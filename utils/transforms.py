"""
Module: transforms.py
Authors: Christian Bergler, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 21.12.2021
"""
import copy
import io
import os
import sys
import math

import ltfatpy
import resampy
import numpy as np
import scipy.fftpack
import soundfile as sf
import pandas as pd

import torch
import torch.nn.functional as F

from typing import List
from multiprocessing import Lock

from matplotlib import pyplot as plt
from torch import Tensor

from utils.constants import *
from utils.FileIO import AsyncFileReader, AsyncFileWriter
from utils.gabor_tools import get_hop_for_time_steps
from utils.naming import OrcaSpotNameGenerator
from utils.parameter import DataParameters
from utils.phase import pghi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def display_tensor(d, title=None, transpose=True, show_tensor=False, hist=True, spec=True):
    if not show_tensor:
        return

    def _spec(t):
        plt.imshow(t, origin="lower", interpolation=None)
        if title is not None:
            plt.title(title)
        plt.show()

    def _hist(t):
        s = torch.flatten(t).numpy()
        plt.hist(s)
        plt.axvline(np.round(np.min(s), decimals=2))
        plt.text(np.round(np.min(s), decimals=2), 0, f"Min: {np.round(np.min(s), decimals=2)}", rotation=90)
        plt.text(np.round(np.mean(s), decimals=2), 0, f"Mean: {np.round(np.mean(s), decimals=2)}", rotation=90)
        plt.text(np.round(np.max(s), decimals=2), 0, f"Max: {np.round(np.max(s), decimals=2)}", rotation=90)

        if title is not None:
            plt.title(title)
        plt.show()

    def _hist_spec(t):
        figure, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(t, origin="lower", interpolation=None)
        s = torch.flatten(t).numpy()
        axs[1].hist(s)
        axs[1].axvline(np.round(np.min(s), decimals=2))
        axs[1].text(np.round(np.min(s), decimals=2), 0, f"Min: {np.round(np.min(s), decimals=2)}", rotation=90)
        axs[1].text(np.round(np.mean(s), decimals=2), 0, f"Mean: {np.round(np.mean(s), decimals=2)}", rotation=90)
        axs[1].text(np.round(np.max(s), decimals=2), 0, f"Max: {np.round(np.max(s), decimals=2)}", rotation=90)
        if title is not None:
            plt.suptitle(title)
        plt.show()

    if len(d.shape) == 3:
        d = d[0]
    if transpose:
        d = d.T

    if hist and spec:
        _hist_spec(d)
    else:
        if hist:
            _hist(d)
        if spec:
            _spec(d)


def load_audio_file(file_name, sr=None, mono=True):
    """
    Load audio file
    """
    y, sr_orig = sf.read(file_name, always_2d=True, dtype="float32")
    if mono and y.ndim == 2 and y.shape[1] > 1:
        y = np.mean(y, axis=1, keepdims=True)
    if sr is not None and sr != sr_orig:
        y = resampy.resample(y, sr_orig, sr, axis=0, filter="kaiser_best")
    return torch.from_numpy(y).float().t()


class Compose(object):
    """
    Composes several transforms to one.
    """
    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], list):
            self.transforms = transforms[0]
        else:
            self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class SqueezeDim0(object):
    """
    Unsqueezes the given tensor at dim=0.
    """
    def __call__(self, x):
        return x.squeeze(dim=0)


class UnsqueezeDim0(object):
    """
    Unsqueezes the given tensor at dim=0.
    """
    def __call__(self, x):
        return x.unsqueeze(dim=0)


class ToFloatTensor(object):
    """
    Converts a given numpy array to torch.FloatTensor.
    """
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            return x.float()
        else:
            raise ValueError("Unknown input array type: {}".format(type(x)))


class ToFloatNumpy(object):
    """
    Converts a given numpy array to torch.FloatTensor.
    """
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return x.astype("float32")
        elif isinstance(x, torch.Tensor):
            return x.float().numpy()
        else:
            raise ValueError("Unknown input array type: {}".format(type(x)))


class Decompress(object):
    """
    Frequency decompression of a given frequency range into a chosen number of frequency bins (important for
    reconstruction of the cmplx spectrogram).
    """
    def __init__(self, f_min=500, f_max=12500, n_fft=4096, sr=44100):
        self.sr = sr
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, spectrogram):
        min_bin = int(max(0, math.floor(self.n_fft * self.f_min / self.sr)))
        max_bin = int(min(self.n_fft - 1, math.ceil(self.n_fft * self.f_max / self.sr)))

        spec = F.interpolate(spectrogram, size=(spectrogram.size(2), max_bin - min_bin), mode="nearest").squeeze(dim=0)
        lower_spec = torch.zeros([1, spectrogram.size(2), min_bin])
        upper_spec = torch.zeros([1, spectrogram.size(2), (self.n_fft // 2 + 1) - max_bin])

        final_spec = torch.cat((lower_spec, spec), 2)
        final_spec = torch.cat((final_spec, upper_spec), 2)

        return final_spec


class PreEmphasize(object):
    """
    Pre-Emphasize in order to raise higher frequencies and lower low frequencies.
    """
    def __init__(self, factor=0.97):
        self.factor = factor

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "PreEmphasize expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        return torch.cat(
            (y[:, 0].unsqueeze(dim=-1), y[:, 1:] - self.factor * y[:, :-1]), dim=-1
        )


class Spectrogram(object):
    """
    Converts a given audio to a spectrogram.
    """
    def __init__(self, n_fft, hop_length, center=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(self.n_fft)

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "Spectrogram expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        S = torch.stft(
            input=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            onesided=True,
            return_complex=False
        ).transpose(1, 2)
        S /= self.window.pow(2).sum().sqrt()
        S = S.pow(2).sum(-1)
        return S


class CachedSpectrogram(object):
    """
    Converts a given audio to a spectrogram, cache and store the spectrograms.
    """
    version = 4

    def __init__(
        self, cache_dir, spec_transform, file_reader=None, file_writer=None, **meta
    ):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        if file_reader is not None:
            self.reader = file_reader
        else:
            self.reader = AsyncFileReader(n_readers=1)
        self.transform = spec_transform
        self.meta = meta
        if file_writer is not None:
            self.writer = file_writer
        else:
            self.writer = AsyncFileWriter(write_fn=self._write_fn, n_writers=1)

    def get_cached_name(self, file_name):
        cached_spec_n = os.path.splitext(os.path.basename(file_name))[0] + ".spec"
        dir_structure = os.path.dirname(file_name).replace(r"/", "_") + "_"
        cached_spec_n = dir_structure + cached_spec_n
        if not os.path.isabs(cached_spec_n):
            cached_spec_n = os.path.join(self.cache_dir, cached_spec_n)
        return cached_spec_n

    def __call__(self, fn):
        cached_spec_n = self.get_cached_name(fn)
        if not os.path.isfile(cached_spec_n):
            return self._compute_and_cache(fn)
        try:
            data = self.reader(cached_spec_n)
            spec_dict = torch.load(io.BytesIO(data), map_location="cpu")
        except (EOFError, RuntimeError):
            return self._compute_and_cache(fn)
        if not (
            "v" in spec_dict
            and spec_dict["v"] == self.version
            and "data" in spec_dict
            and spec_dict["data"].dim() == 3
        ):
            return self._compute_and_cache(fn)
        for key, value in self.meta.items():
            if not (key in spec_dict and spec_dict[key] == value):
                return self._compute_and_cache(fn)
        return spec_dict["data"]

    def _compute_and_cache(self, fn):
        try:
            audio_data = self.reader(fn)
            spec = self.transform(io.BytesIO(audio_data))
        except Exception:
            spec = self.transform(fn)
        self.writer(self.get_cached_name(fn), spec)
        return spec

    def _write_fn(self, fn, data):
        spec_dict = {"v": self.version, "data": data}
        for key, value in self.meta.items():
            spec_dict[key] = value
        torch.save(spec_dict, fn)


class MeanStdNormalize(object):
    """
    Normalize a spectrogram by subtracting mean and dividing by std.
    """
    def __call__(self, spectrogram, ret_dict=None):
        mean = spectrogram.mean()
        spectrogram.sub_(mean)
        std = spectrogram.std()
        spectrogram.div_(std)
        if ret_dict is not None:
            ret_dict["mean"] = mean
            ret_dict["std"] = std
        return spectrogram


class Normalize(object):
    """
    Normalize db scale to 0..1
    """
    def __init__(self, min_level_db=-100, ref_level_db=20):
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

    def __call__(self, spec):
        return torch.clamp(
            (spec - self.ref_level_db - self.min_level_db) / -self.min_level_db, 0, 1
        )


class MinMaxNormalize(object):
    """
    Normalize min/max scale to 0..1
    """
    def __call__(self, spectrogram):
        spectrogram -= spectrogram.min()
        if spectrogram.max().item() == 0.0:
            return spectrogram
        spectrogram /= spectrogram.max()
        return spectrogram


class Amp2Db(object):
    """
    Turns a spectrogram from the power/amplitude scale to the decibel scale.

    Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py

    BSD 2-Clause License

    Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Access Data: 12.09.2018, Last Access Date: 21.12.2021
    Changes: Modified by Christian Bergler and Hendrik Schroeter (12.09.2018)
    """
    def __init__(self, min_level_db=None, stype="power"):
        self.stype = stype
        self.multiplier = 10. if stype == "power" else 20.
        if min_level_db is None:
            self.min_level = None
        else:
            min_level_db = -min_level_db if min_level_db > 0 else min_level_db
            self.min_level = torch.tensor(
                np.exp(min_level_db / self.multiplier * np.log(10)), dtype=torch.float
            )

    def __call__(self, spec):
        if self.min_level is not None:
            spec_ = torch.max(spec, self.min_level)
        else:
            spec_ = spec
        spec_db = self.multiplier * torch.log10(spec_)
        return spec_db


class SPECLOG1P(object):
    """
    Compress a spectrogram using torch.log1p(spec * compression_factor).
    """
    def __init__(self, compression_factor=1):
        self.compression_factor = compression_factor

    def __call__(self, spectrogram):
        return torch.log1p(spectrogram * self.compression_factor)


class SPECEXPM1(object):
    """
    Compress a spectrogram using torch.log1p(spec * compression_factor).
    """
    def __init__(self, decompression_factor=1):
        self.decompression_factor = decompression_factor

    def __call__(self, spectrogram):
        return torch.expm1(spectrogram) / self.decompression_factor


def _scale(spectrogram: torch.Tensor, shift_factor: float, dim: int):
    """
    Scaling spectrogram dimension (time/frequency) by a given factor.
    """
    in_dim = spectrogram.dim()
    if in_dim < 3:
        raise ValueError(
            "Expected spectrogram with size (c t f) or (n c t f)"
            ", but got {}".format(spectrogram.size())
        )
    if in_dim == 3:
        spectrogram.unsqueeze_(dim=0)
    size = list(spectrogram.shape)[2:]
    dim -= 1
    size[dim] = int(round(size[dim] * shift_factor))
    spectrogram = F.interpolate(spectrogram, size=size, mode="nearest")
    if in_dim == 3:
        spectrogram.squeeze_(dim=0)
    return spectrogram


class RandomPitchShift(object):
    """
    Randomly shifts the pitch of a spectrogram by a factor of 2**Uniform(log2(from), log2(to)).
    """
    def __init__(self, from_=1, to_=1.8):
        self.from_ = math.log2(from_)
        self.to_ = math.log2(to_)

    def __call__(self, spectrogram: torch.Tensor):
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item()
        median = spectrogram.median()
        size = list(spectrogram.shape)
        scaled = _scale(spectrogram, factor, dim=2)
        if factor > 1:
            out = scaled[:, :, : size[2]]
        else:
            out = torch.full(size, fill_value=median, dtype=spectrogram.dtype)
            new_f_bins = int(round(size[2] * factor))
            out[:, :, 0:new_f_bins] = scaled
        return out


class RandomTimeStretch(object):
    """
    Randomly stretches the time of a spectrogram by a factor of 2**Uniform(log2(from), log2(to)).
    """
    def __init__(self, from_=0.5, to_=0.7):
        self.from_ = math.log2(from_)
        self.to_ = math.log2(to_)

    def __call__(self, spectrogram: torch.Tensor):
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item()
        return _scale(spectrogram, factor, dim=1)


class RandomAmplitude(object):
    """
    Randomly scaling (uniform distributed) the amplitude based on a given input spectrogram (intensity augmenation).
    """
    def __init__(self, increase_db=3, decrease_db=None, device=None):
        self.inc_db = increase_db
        if decrease_db is None:
            decrease_db = -increase_db
        elif decrease_db > 0:
            decrease_db *= -1
        self.dec_db = decrease_db
        self.device = device

    def __call__(self, spec):
        db_change = torch.randint(
            self.dec_db, self.inc_db, size=(1,), dtype=torch.float, device=self.device
        )
        spec = spec.to(self.device)
        return spec.mul(10 ** (db_change / 10))


class RandomAddNoise(object):
    """
    Randomly adds a given noise file to the given spectrogram by considering a randomly selected
    (uniform distributed) SNR of min = -3 dB and max = 12 dB. The noise file could also be intensity, pitch, and/or time
    augmented. If a noise file is longer/shorter than the given spectrogram it will be subsampled/self-concatenated.
    The spectrogram is expected to be a power spectrogram, which is **not** logarithmically compressed.
    """
    def __init__(
        self,
        noise_files: List[str],
        spectrogram_transform,
        transform,
        min_length=0,
        min_snr=12,
        max_snr=5,
        return_original=False,
        device=None
    ):
        if not noise_files:
            raise ValueError("No noise files found")
        self.noise_files = noise_files
        self.t_spectrogram = spectrogram_transform
        self.noise_file_locks = {file: Lock() for file in noise_files}
        self.transform = transform
        self.min_length = min_length
        self.t_pad = PaddedSubsequenceSampler(sequence_length=min_length, dim=1)
        self.min_snr = min_snr if min_snr > max_snr else max_snr
        self.max_snr = max_snr if min_snr > max_snr else min_snr
        self.return_original = return_original
        self.device = device

    def __call__(self, spectrogram):
        if len(self.noise_files) == 1:
            idx = 0
        else:
            idx = torch.randint(
                0, len(self.noise_files) - 1, size=(1,), dtype=torch.long
            ).item()
        noise_file = self.noise_files[idx]

        try:
            if not self.noise_file_locks[noise_file].acquire(timeout=10):
                print("Warning: Could not acquire lock for {}".format(noise_file))
                return spectrogram
            noise_spec = self.t_spectrogram(noise_file)
        except Exception:
            import traceback

            print(traceback.format_exc())
            return spectrogram
        finally:
            self.noise_file_locks[noise_file].release()

        noise_spec = self.t_pad._maybe_sample_subsequence(
            noise_spec, spectrogram.size(1) * 2
        )
        noise_spec = self.transform(noise_spec)

        if self.min_length > 0:
            spectrogram = self.t_pad._maybe_pad(spectrogram)

        if spectrogram.size(1) > noise_spec.size(1):
            n_repeat = int(math.ceil(spectrogram.size(1) / noise_spec.size(1)))
            noise_spec = noise_spec.repeat(1, n_repeat, 1)
        if spectrogram.size(1) < noise_spec.size(1):
            high = noise_spec.size(1) - spectrogram.size(1)
            start = torch.randint(0, high, size=(1,), dtype=torch.long)
            end = start + spectrogram.size(1)
            noise_spec_part = noise_spec[:, start:end]
        else:
            noise_spec_part = noise_spec

        snr = torch.randint(self.max_snr, self.min_snr, size=(1,), dtype=torch.float, device=self.device)
        signal_power = spectrogram.sum()
        noise_power = noise_spec_part.sum()

        noise_spec_part = noise_spec_part[:, :spectrogram.shape[1], :spectrogram.shape[2]]

        K = (signal_power / noise_power) * 10 ** (-snr / 10)
        if self.device is not None:
            noise_spec_part = noise_spec_part.to(self.device)
            spectrogram = spectrogram.to(self.device)
            K = K.to(self.device)

        spectrogram_aug = spectrogram + noise_spec_part * K

        if self.return_original:
            return spectrogram_aug, spectrogram
        return spectrogram_aug


class PaddedSubsequenceSampler(object):
    """
    Samples a subsequence along one axis and pads if necessary.
    """
    def __init__(self, sequence_length: int, dim: int = 0, random=True, device=None):
        assert isinstance(sequence_length, int)
        assert isinstance(dim, int)
        self.sequence_length = sequence_length
        self.dim = dim
        self.device = device
        if random:
            self._sampler = lambda x: torch.randint(
                0, x, size=(1,), dtype=torch.long
            ).item()
        else:
            self._sampler = lambda x: x // 2

    def _maybe_sample_subsequence(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length > sequence_length:
            start = self._sampler(sample_length - sequence_length)
            end = start + sequence_length
            indices = torch.arange(start, end, dtype=torch.long, device=self.device)
            return torch.index_select(spectrogram.to(self.device), self.dim, indices)
        return spectrogram

    def _maybe_pad(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length < sequence_length:
            start = self._sampler(sequence_length - sample_length)
            end = start + sample_length

            shape = list(spectrogram.shape)
            shape[self.dim] = sequence_length
            padded_spectrogram = torch.zeros(shape, dtype=spectrogram.dtype)

            if self.dim == 0:
                padded_spectrogram[start:end] = spectrogram
            elif self.dim == 1:
                padded_spectrogram[:, start:end] = spectrogram
            elif self.dim == 2:
                padded_spectrogram[:, :, start:end] = spectrogram
            elif self.dim == 3:
                padded_spectrogram[:, :, :, start:end] = spectrogram
            return padded_spectrogram
        return spectrogram

    def __call__(self, spectrogram):
        spectrogram = self._maybe_pad(spectrogram)
        spectrogram = self._maybe_sample_subsequence(spectrogram)
        return spectrogram


class Interpolate(object):
    """
    Frequency compression of a given frequency range into a chosen number of frequency bins.
    """
    def __init__(self, n_freqs, sr=None, f_min=0, f_max=None):
        self.n_freqs = n_freqs
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, spec):
        n_fft = (spec.size(2) - 1) * 2

        if self.sr is not None and n_fft is not None:
            min_bin = int(max(0, math.floor(n_fft * self.f_min / self.sr)))
            max_bin = int(min(n_fft - 1, math.ceil(n_fft * self.f_max / self.sr)))
            spec = spec[:, :, min_bin:max_bin]

        spec.unsqueeze_(dim=0)
        spec = F.interpolate(spec, size=(spec.size(2), self.n_freqs), mode="nearest")
        return spec.squeeze(dim=0)


def _hz2mel(f):
    """
    Convert hertz to mel.
    """
    return 2595 * np.log10(1 + f / 700)


def _mel2hz(mel):
    """
    Convert mel to hertz.
    """
    return 700 * (10 ** (mel / 2595) - 1)


def _melbank(sr, n_fft, n_mels=128, f_min=0.0, f_max=None, inverse=False):
    """
    Create melbank.
    Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py

    BSD 2-Clause License

    Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Access Data: 12.09.2018, Last Access Date: 21.12.2021
    Changes: Modified by Christian Bergler and Hendrik Schroeter (12.09.2018)
    """
    m_min = 0. if f_min == 0 else _hz2mel(f_min)
    m_max = _hz2mel(f_max if f_max is not None else sr // 2)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel2hz(m_pts)

    bins = torch.floor(((n_fft - 1) * 2 + 1) * f_pts / sr).long()

    fb = torch.zeros(n_mels, n_fft)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1].item()
        f_m = bins[m].item()
        f_m_plus = bins[m + 1].item()

        if f_m_minus != f_m:
            fb[m - 1, f_m_minus:f_m] = (torch.arange(f_m_minus, f_m) - f_m_minus).float() / (
                f_m - f_m_minus
            )
        if f_m != f_m_plus:
            fb[m - 1, f_m:f_m_plus] = (f_m_plus - torch.arange(f_m, f_m_plus)).float() / (
                f_m_plus - f_m
            )

    if not inverse:
        return fb.t()
    else:
        return fb


class F2M(object):
    """
    This turns a normal STFT into a MEL Frequency STFT, using a conversion matrix.  This uses triangular filter banks.
    Code from https://github.com/pytorch/audio/blob/5787787edc/torchaudio/transforms.py

    BSD 2-Clause License

    Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Access Data: 12.09.2018, Last Access Date: 21.12.2021
    Changes: Modified by Christian Bergler and Hendrik Schroeter (12.09.2018)
    """

    def __init__(
        self, sr: int = 16000, n_mels: int = 40, f_min: int = 0, f_max: int = None
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_f: torch.Tensor):
        n_fft = spec_f.size(2)

        fb = _melbank(self.sr, n_fft, self.n_mels, self.f_min, self.f_max)

        spec_m = torch.matmul(
            spec_f.double(), fb.double()
        )
        return spec_m


class M2F(object):
    """
    Converts a normal STFT into a MEL Frequency STFT, using a conversion
    matrix. This uses triangular filter banks.
    """
    def __init__(
        self, sr: int = 16000, n_fft: int = 1024, f_min: int = 0, f_max: int = None
    ):
        self.sr = sr
        self.n_fft = n_fft // 2 + 1
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_m: torch.Tensor):
        n_mels = spec_m.size(2)

        fb = _melbank(self.sr, self.n_fft, n_mels, self.f_min, self.f_max, inverse=True)

        spec_f = torch.matmul(
            spec_m, fb
        )
        return spec_f


class M2MFCC(object):
    """
    Converts MEL Frequency to MFCC.
    """
    def __init__(self, n_mfcc : int = 32):
        self.n_mfcc = n_mfcc

    def __call__(self, spec_m):
        device = spec_m.device
        spec_m = 10 * torch.log10(spec_m)
        spec_m[spec_m == float('-inf')] = 0
        if isinstance(spec_m, torch.Tensor):
            spec_m = spec_m.cpu().numpy()
        mfcc = scipy.fftpack.dct(spec_m, axis=-1)
        mfcc = mfcc[:, :, 1:self.n_mfcc+1]
        return torch.from_numpy(mfcc).to(device)


class GaborSpectrogramPreparation:
    def __init__(self,
                 normalize=True,
                 clip_below_factor=-10,
                 frequency_bins=512,
                 sequence_length=256,
                 log=True,
                 clip=True,
                 debug=False):
        self.normalize = normalize
        self.clip_below_factor = clip_below_factor
        self.clip_below = np.e ** clip_below_factor
        self.frequency_bins = frequency_bins
        self.sequence_length = sequence_length
        self.log = log
        self.clip = clip
        self.debug = debug

    def __call__(self, abs_dgt, compose_prep=True) -> Tensor:
        spectrogram = copy.deepcopy(abs_dgt)
        if type(abs_dgt) != Tensor:
            spectrogram = torch.from_numpy(spectrogram)

        if self.normalize:
            spectrogram /= torch.max(spectrogram)
            display_tensor(spectrogram, title="Normalized", show_tensor=self.debug, transpose=False)
        if self.clip:
            spectrogram = torch.clip(spectrogram, min=self.clip_below, max=None)
            display_tensor(spectrogram, title="Clipped", show_tensor=self.debug, transpose=False)
        if self.log:
            spectrogram = torch.log(spectrogram)
            display_tensor(spectrogram, title="Logged", show_tensor=self.debug, transpose=False)
        if compose_prep:
            spectrogram = spectrogram.T.unsqueeze(0)
        spectrogram = spectrogram / (-self.clip_below_factor / 2) + 1
        display_tensor(spectrogram, title="Normalized Again", show_tensor=self.debug, transpose=True)
        return spectrogram

class GaborNormalization:
    def __init__(self,
                 normalize=True,
                 clip_below_factor=-10,
                 frequency_bins=512,
                 sequence_length=256,
                 log=True,
                 clip=True,
                 debug=False):
        self.normalize = normalize
        self.clip_below_factor = clip_below_factor
        self.clip_below = np.e ** clip_below_factor
        self.frequency_bins = frequency_bins
        self.sequence_length = sequence_length
        self.log = log
        self.clip = clip
        self.debug = debug

    def __call__(self, abs_dgt, compose_prep=False) -> Tensor:
        spectrogram = copy.deepcopy(abs_dgt)
        if type(abs_dgt) != Tensor:
            spectrogram = torch.from_numpy(spectrogram)

        if self.normalize:
            spectrogram /= torch.max(spectrogram)
            display_tensor(spectrogram, title="Normalized", show_tensor=self.debug, transpose=False)
        if self.clip:
            spectrogram = torch.clip(spectrogram, min=self.clip_below, max=None)
            display_tensor(spectrogram, title="Clipped", show_tensor=self.debug, transpose=False)
        if self.log:
            spectrogram = torch.log(spectrogram)
            display_tensor(spectrogram, title="Logged", show_tensor=self.debug, transpose=False)
        if compose_prep:
            spectrogram = spectrogram.T.unsqueeze(0)
        spectrogram = spectrogram / (-self.clip_below_factor / 2) + 1
        display_tensor(spectrogram, title="Normalized Again", show_tensor=self.debug, transpose=True)

        return spectrogram

class GaborSpectrogram:
    def __init__(self,
                 n_fft,
                 hop_length,
                 normalize=True,
                 clip_below_factor=-10,
                 frequency_bins=512,
                 sequence_length=256,
                 log=True,
                 clip=True,
                 prep_for_compose=True,
                 apply_preparation=True,
                 debug=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.clip_below_factor = clip_below_factor
        self.clip_below = np.e ** clip_below_factor
        self.frequency_bins = (n_fft // 2) + 1
        self.sequence_length = sequence_length
        self.log = log
        self.clip = clip
        self.compose_prep = prep_for_compose
        self.window = {
            'name': 'gauss',
            'M': n_fft
        }
        self.debug = debug

        self.apply_prep = apply_preparation
        self.prep = GaborSpectrogramPreparation(normalize=self.normalize,
                                                clip_below_factor=clip_below_factor,
                                                frequency_bins=self.frequency_bins,
                                                sequence_length=self.sequence_length,
                                                log=self.log,
                                                clip=self.clip,
                                                debug=self.debug)

    def __call__(self, signal):
        if type(signal) == Tensor:
            signal = signal.numpy().astype('float64')
        if len(signal.shape) == 2:
            signal = signal[0, :]
        if self.hop_length == dynamic:
            hop_len = get_hop_for_time_steps(
                signal_length=len(signal),
                n_fft=self.n_fft,
                time_steps=self.sequence_length)
        else:
            hop_len = self.hop_length
        dgt, a, b = ltfatpy.dgtreal(signal, self.window, hop_len, self.n_fft)

        spectrogram = np.abs(dgt)
        if self.apply_prep:
            spectrogram = self.prep(spectrogram)
        else:
            spectrogram = torch.from_numpy(spectrogram).T.unsqueeze(0)

        display_tensor(spectrogram, title="Sample", show_tensor=self.debug, transpose=True)

        return spectrogram


class InverseGaborSpectrogram:
    def __init__(self,
                 clip_factor):
        self.clip_below_factor = clip_factor

    def __call__(self, spectrogram, show_tensor=False):
        copied_spectrogram = copy.deepcopy(spectrogram)
        copied_spectrogram = np.exp((-self.clip_below_factor / 2) * (copied_spectrogram - 1))
        display_tensor(copied_spectrogram, title="Inversed Spectrogram step 1", show_tensor=show_tensor, transpose=True)

        return copied_spectrogram

class AudioThresholdSlice:
    def __init__(self, threshold=0.002):
        self.threshold = threshold

    def __call__(self, audio_array, start=None, end=None):
        if start is None or end is None:
            audio_array[audio_array <= self.threshold] = 0
            idxs = np.nonzero(audio_array)
            if start is None and end is None:
                start = np.min(idxs)
                end = np.max(idxs)
            elif start is None and end is not None:
                start = np.min(idxs)
            elif start is not None and end is None:
                end = np.max(idxs)
        audio_slice = audio_array[start: end]
        return audio_slice


class GaborInverter:
    def __init__(self,
                 data_parameters: DataParameters,
                 name_generator: OrcaSpotNameGenerator = None,
                 output_dir: str = None,
                 class_name: str = "orca",
                 device=None,
                 debug=False
                 ):
        self.data_parameters = data_parameters
        self.frequency_first = data_parameters.frequency_first
        self.name_generator = name_generator
        self.output_dir = output_dir
        self.n_fft = data_parameters.n_fft
        self.hop_length = data_parameters.inversion.inversion_hop_length
        self.logged_input = data_parameters.log_input
        self.exp_power = data_parameters.inversion.exp_power
        self.sample_rate = data_parameters.sr
        self.class_name = class_name
        self.device = device

        self.anStftWrapper = LTFATStft()
        self.slicer = self._get_slicer()
        self.slice_start = None
        self.slice_end = None
        self.post_processing_t = self._get_post_processing_t()
        self.debug = debug

        if self.data_parameters.frequency_compression:
            self.decompressor = Decompress(
                f_min=self.data_parameters.frequency_compression.f_min,
                f_max=self.data_parameters.frequency_compression.f_max,
                n_fft=self.n_fft,
                sr=self.sample_rate
            )
        else:
            self.decompressor = Decompress(
                f_min=0,
                f_max=self.sample_rate // 2,
                n_fft=self.n_fft,
                sr=self.sample_rate
            )

    def _get_slicer(self):
        if not self.data_parameters.inversion.post_processing.slice.active:
            return None
        else:
            return AudioThresholdSlice(
                threshold=self.data_parameters.inversion.post_processing.slice.threshold
            )

    def _get_post_processing_t(self):
        post_processing_t = []
        if self.data_parameters.inversion.post_processing.time_stretch.active:
            post_processing_t.append(
                RandomTimeStretch(
                    from_=self.data_parameters.inversion.post_processing.time_stretch.from_,
                    to_=self.data_parameters.inversion.post_processing.time_stretch.to_
                )
            )
        if self.data_parameters.inversion.post_processing.sampler.active:
            post_processing_t.append(
                PaddedSubsequenceSampler(
                    sequence_length=self.data_parameters.inversion.post_processing.sampler.sequence_length,
                    dim=self.data_parameters.inversion.post_processing.sampler.dim,
                    random=self.data_parameters.inversion.post_processing.sampler.random,
                    device=self.device
                )
            )
        if len(post_processing_t) > 0:
            return Compose(post_processing_t)
        return None

    def _sample_hop_length(self):
        hop_file = os.path.join(self.data_parameters.data_directory, ".hop.csv")
        if not os.path.isfile(hop_file):
            print(f"Hop file for dynamic hop lengths not found at {hop_file}")
            exit(1)
        df = pd.read_csv(hop_file)
        if self.hop_length == dynamic:

            hop = df["inversion_hop"].sample(n=1).item()
        else:
            hop = int(df["inversion_hop"].median())
        return hop


    def _prepare(self, spectrogram):
        if type(spectrogram) == Tensor:
            spectrogram = spectrogram.detach().numpy()
        if self.logged_input:
            spectrogram = np.exp(self.exp_power * (spectrogram - 1))

        if spectrogram.shape[0] != (self.n_fft // 2) + 1:
            spectrogram = np.concatenate([spectrogram, np.zeros_like(spectrogram)[0:1, :]], axis=0)
        return spectrogram

    def _reconstruct(self, spectrogram):
        # Compute Tgrad and Fgrad from the generated spectrograms
        if self.hop_length == dynamic or self.hop_length == "median":
            hop = self._sample_hop_length()
        else:
            hop = self.hop_length
        L = hop * self.data_parameters.inversion.inversion_time_bins
        gs = {'name': 'gauss', 'M': self.n_fft}

        t_grad, f_grad = ltfatpy.gabphasegrad('abs', spectrogram, gs, hop)
        if self.logged_input:
            spectrogram = np.log(spectrogram.astype(np.float64))
        phase = pghi(spectrogram, t_grad, f_grad, hop, self.n_fft, L, tol=10)
        reconstructed_audio = self.anStftWrapper.reconstructSignalFromLoggedSpectogram(spectrogram, phase,
                                                                                       windowLength=self.n_fft,
                                                                                       hopSize=hop)
        return reconstructed_audio

    def _decompress(self, spectrogram, already_transposed=False):
        if self.frequency_first and not already_transposed:
            spectrogram = spectrogram.transpose(1, 2)
        if len(spectrogram.shape) == 3:
            spectrogram = torch.unsqueeze(spectrogram, dim=0)
        spectrogram = self.decompressor(spectrogram)
        spectrogram = spectrogram.transpose(1, 2)
        return spectrogram

    def _write_audio(self, audio_array):
        output_file = os.path.join(self.output_dir,
                                   self.name_generator.generate(self.class_name).replace(".png", ".wav"))

        if self.debug:
            print(f"{len(audio_array) / self.sample_rate}s of audio written to {output_file}")
        sf.write(output_file, audio_array, self.sample_rate)

    def _set_slice_points(self, spectrogram, already_transposed=False):
        if self.frequency_first and not already_transposed:
            time_dim = spectrogram[0, 0, :]
        else:
            time_dim = spectrogram[0, :, 0]
        time_dim = time_dim.detach().numpy().nonzero()
        start = np.min(time_dim)
        end = np.max(time_dim)
        start_sample = start * self.hop_length
        end_sample = end * self.hop_length
        self.slice_start = start_sample
        self.slice_end = end_sample

    def __call__(self, spectrogram):
        already_transposed = False
        if self.post_processing_t is not None:
            if self.frequency_first:
                spectrogram = spectrogram.transpose(1, 2)
                already_transposed = True
            spectrogram = self.post_processing_t(spectrogram)
        else:
            if self.frequency_first:
                spectrogram = spectrogram.transpose(1, 2)
                already_transposed = True
        display_tensor(spectrogram, "Inverter Input", transpose=True, show_tensor=self.debug)
        self._set_slice_points(spectrogram, already_transposed)
        spectrogram = self._decompress(spectrogram, already_transposed)
        spectrogram = spectrogram[0, :, :]
        reconstructed = self._reconstruct(self._prepare(spectrogram))
        if self.slicer is not None:
            reconstructed = self.slicer(reconstructed, self.slice_start, self.slice_end)
        if self.output_dir is not None and self.name_generator is not None:
            self._write_audio(reconstructed)
        return reconstructed


class LTFATStft(object):
    """
    Tools for dealing with Gabor Transforms.
    Code from https://github.com/tifgan/stftGAN/blob/master/data/ourLTFATStft.py

    GNU3 License

    All rights reserved.

    Access Data: 03.17.2022, Last Access Date: 03.17.2022
    """
    def oneSidedStft(self, signal, windowLength, hopSize):
        gs = {'name': 'gauss', 'M': windowLength}
        return ltfatpy.dgtreal(signal, gs, hopSize, windowLength)[0]

    def inverseOneSidedStft(self, signal, windowLength, hopSize):
        synthesis_window = {'name': 'gauss', 'M': windowLength}
        analysis_window = {'name': ('dual', synthesis_window['name']), 'M': synthesis_window['M']}

        return ltfatpy.idgtreal(signal, analysis_window, hopSize, windowLength)[0]

    def magAndPhaseOneSidedStft(self, signal, windowLength, hopSize):
        stft = self.oneSidedStft(signal, windowLength, hopSize)
        return np.abs(stft), np.angle(stft)

    def log10MagAndPhaseOneSidedStft(self, signal, windowLength, hopSize, clipBelow=1e-14):
        realDGT = self.oneSidedStft(signal, windowLength, hopSize)
        return self.log10MagFromRealDGT(realDGT, clipBelow), np.angle(realDGT)

    def log10MagFromRealDGT(self, realDGT, clipBelow=1e-14):
        return np.log10(np.clip(np.abs(realDGT), a_min=clipBelow, a_max=None))

    def reconstructSignalFromLogged10Spectogram(self, logSpectrogram, phase, windowLength, hopSize):
        reComplexStft = (10 ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft, windowLength, hopSize)

    def logMagAndPhaseOneSidedStft(self, signal, windowLength, hopSize, clipBelow=np.e**-30, normalize=False):
        realDGT = self.oneSidedStft(signal, windowLength, hopSize)
        spectrogram = self.logMagFromRealDGT(realDGT, clipBelow, normalize)
        return spectrogram, np.angle(realDGT)

    def logMagFromRealDGT(self, realDGT, clipBelow=np.e**-30, normalize=False):
        spectrogram = np.abs(realDGT)
        if normalize:
            spectrogram = spectrogram/np.max(spectrogram)
        return np.log(np.clip(spectrogram, a_min=clipBelow, a_max=None))

    def reconstructSignalFromLoggedSpectogram(self, logSpectrogram, phase, windowLength, hopSize):
        reComplexStft = (np.e ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft, windowLength, hopSize)


if __name__ == '__main__':
    test_file = "/media/alex/Datasets/MONK-PARAKEET/monk_call_type_less_noise/contact/contact-contact_4725_2019_2019-11-19-173258_224302_224591.wav"
    sr = 44100
    n_fft = 1024
    hop_length = 100
    spec_transforms = [
        lambda fn: load_audio_file(fn, sr=sr),
        PreEmphasize(),
        GaborSpectrogram(n_fft, hop_length, debug=True),
    ]
    g_spec = Compose(spec_transforms)
    spec = g_spec(test_file)
    print(spec.shape)