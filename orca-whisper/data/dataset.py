import os
import shutil

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import utils.transforms as T
from torch import Tensor
import matplotlib.pyplot as plt

import random
import numpy as np
import soundfile as sf
import pandas as pd

from utils.gabor_tools import get_hop_for_time_steps, get_signal_0_padding, get_adjusted_signal_length
from utils.logging import Logger
from utils.parameter import DataParameters, CacheParameters, Parameters
from utils.constants import *

DefaultSpecDatasetOps = {
    "sr": 44100,
    "preemphases": 0.98,
    "n_fft": 4096,
    "hop_length": 441,
    "n_freq_bins": 256,
    "fmin": 500,
    "fmax": 10000,
    "freq_compression": "linear",
    "min_level_db": -100,
    "ref_level_db": 20,
    "freq_first": True
}


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


class GaborCache(object):
    def __init__(self, cache_parameters: CacheParameters):
        self.base_dir = cache_parameters.cache_directory
        self.bust = cache_parameters.bust_cache
        self.img = cache_parameters.create_img
        self.ext = cache_parameters.target_ext

        self.cache_dir = os.path.join(self.base_dir, ".cache")
        if self.bust and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if self.img:
            self.img_dir = os.path.join(self.cache_dir, ".img")
            if not os.path.isdir(self.img_dir):
                os.makedirs(self.img_dir)

    def _write_img(self, file_name, tensor, source_ext):
        figure, ax = plt.subplots(dpi=300)
        if len(tensor.shape) == 3:
            tensor = tensor[0]
        if tensor.shape[1] > tensor.shape[0]:
            tensor = tensor.T

        plt.imshow(tensor, origin="lower", interpolation=None)

        plt.savefig(os.path.join(self.img_dir, os.path.basename(file_name).replace(source_ext, "png")),
                    bbox_inches="tight")
        plt.close(figure)

    def __call__(self, original_file_name, tensor=None) -> Tensor or None:
        """
        Get cached entry if tensor is None, else save it
        Parameters
        ----------
        original_file_name
        tensor

        Returns
        -------

        """
        source_ext = os.path.basename(original_file_name).split(".")[-1]
        cached_item = os.path.join(self.cache_dir, os.path.basename(original_file_name).replace(source_ext, self.ext))
        if tensor is not None:
            torch.save(tensor, cached_item)
            if self.img:
                self._write_img(original_file_name, tensor, source_ext)
        elif os.path.isfile(cached_item):
            return torch.load(cached_item)
        else:
            return None


class GaborPreparation(object):
    def __init__(self,
                 data_parameters: DataParameters,
                 device=None,
                 cache: GaborCache = None,
                 debug=False
                 ):
        self.parameters = data_parameters
        self.cache = cache
        self.n_fft = data_parameters.n_fft
        self.hop_length = data_parameters.hop_length
        self.f_min = data_parameters.frequency_compression.f_min
        self.f_max = data_parameters.frequency_compression.f_max
        self.log_input = data_parameters.log_input
        self.sr = data_parameters.sr
        self.device = device
        self.sequence_length = data_parameters.n_time_bins
        self.n_freq_bins = data_parameters.frequency_compression.n_freq_bins
        self.freq_first = DefaultSpecDatasetOps["freq_first"]
        self.clip_below_factor = data_parameters.clip_below_factor
        self.debug = debug

        self.g_spec_transforms = [
            lambda fn: T.load_audio_file(fn, sr=self.sr),
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.GaborSpectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                debug=self.debug
            ),
        ]

        self.g_spectrogram = T.Compose(self.g_spec_transforms)

        self.freq_compression = data_parameters.frequency_compression.type
        if self.freq_compression == "linear":
            self.g_compr_f = T.Interpolate(
                self.n_freq_bins, self.sr, self.f_min, self.f_max
            )

        elif self.freq_compression == "mel":
            self.g_compr_f = T.F2M(sr=self.sr, n_mels=self.n_freq_bins, f_min=self.f_min, f_max=self.f_max)
        elif self.freq_compression == "mfcc":
            self.g_compr_f = T.Compose(
                T.F2M(sr=self.sr, n_mels=self.n_freq_bins, f_min=self.f_min, f_max=self.f_max)
            )
            self.t_compr_mfcc = T.M2MFCC(n_mfcc=32)

        self.g_subseq = T.PaddedSubsequenceSampler(self.sequence_length, dim=1, random=True, device=device)

    def _load_wav(self, file_name) -> Tensor:
        if self.cache is not None:
            spectrogram = self.cache(file_name)
            if spectrogram is None:
                spectrogram = self.g_spectrogram(file_name)
                self.cache(file_name, spectrogram)
        else:
            spectrogram = self.g_spectrogram(file_name)
        return spectrogram

    def __call__(self, file_name):
        tensor = self._load_wav(file_name)
        if self.parameters.frequency_compression.active:
            tensor = self.g_compr_f(tensor)
            display_tensor(tensor, "frequency_compression", show_tensor=self.debug)

        tensor = self.g_subseq(tensor)

        return tensor


class DynamicHopData:
    def __init__(self,
                 data_directory,
                 data_files,
                 parameters: DataParameters):
        self.data_directory = data_directory
        self.data_files = data_files
        self.output_file = os.path.join(self.data_directory, ".hop.csv")
        self.parameters = parameters

        self.hop_data = {
            "file_name": [],
            "file_sr": [],
            "file_size": [],
            "target_length": [],
            "hop": [],
            "padding": [],
            "inversion_hop": [],
            "inversion_bins": [],
            "inversion_length": [],
        }
        for file in tqdm(data_files):
            d, sr = sf.read(file)
            if len(d.shape) == 2:
                d = np.mean(d, axis=1)
            self.hop_data["file_name"].append(file)
            self.hop_data["file_sr"].append(sr)
            self.hop_data["file_size"].append(len(d))
            self.hop_data["target_length"].append(self.parameters.n_time_bins)
            hop = get_hop_for_time_steps(signal_length=len(d),
                                         n_fft=self.parameters.n_fft,
                                         time_steps=self.parameters.n_time_bins)
            self.hop_data["hop"].append(hop)
            self.hop_data["padding"].append(get_signal_0_padding(signal_length=len(d),
                                                                 n_fft=self.parameters.n_fft,
                                                                 hop_length=hop))
            i_hop = get_hop_for_time_steps(signal_length=len(d),
                                           n_fft=self.parameters.n_fft,
                                           time_steps=self.parameters.inversion.inversion_time_bins)
            self.hop_data["inversion_hop"].append(i_hop)
            self.hop_data["inversion_bins"].append(self.parameters.inversion.inversion_time_bins)
            self.hop_data["inversion_length"].append(get_adjusted_signal_length(signal_length=len(d),
                                                                                n_fft=self.parameters.n_fft,
                                                                                hop_length=i_hop))
        df = pd.DataFrame(self.hop_data)
        df.to_csv(self.output_file, index=False)


class GaborDataset(Dataset):
    def __init__(self,
                 parameters: DataParameters,
                 device,
                 log: Logger,
                 cache: GaborCache = None,
                 ):
        self.data_dir = parameters.data_directory
        self.cache = cache
        self.n_fft = parameters.n_fft
        self.hop_length = parameters.hop_length
        self.sr = parameters.sr
        self.log_input = parameters.log_input
        self.device = device
        self.freq_first = parameters.frequency_first
        self.debug = parameters.debug

        self.log = log

        self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if
                      os.path.isfile(os.path.join(self.data_dir, f)) and (f.endswith(".wav") or f.endswith(cache.ext))]

        if parameters.hop_length == dynamic:
            log.info("Dynamic Hop Selected. Initializing Data")
            DynamicHopData(
                data_directory=self.data_dir,
                data_files=self.files,
                parameters=parameters
            )

        self.prep = GaborPreparation(
            data_parameters=parameters,
            cache=cache,
            device=device
        )

    def __getitem__(self, item):
        file_name = self.files[item]
        spectrogram = self.prep(file_name).transpose(1, 2)
        return spectrogram.to(self.device)

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    parameter_file = "../parameters.json"
    test_file = "/media/alex/Datasets/orca/call_types/N1/extracted_wav/n1_1_1985_158B_49369950_49546350.wav"
    parameters = Parameters(parameter_file)
    prep = GaborPreparation(
        data_parameters=parameters.data,
        cache=None,
        device=torch.device("cpu"),
        debug=True
    )
    prepped = prep(test_file)
    print("")

