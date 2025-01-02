from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt

from utils.parameter import DataParameters
from utils.transforms import GaborInverter


class Metric:
    def __init__(self, writer: SummaryWriter, name):
        self.writer = writer
        self.name = name

    def update(self, data, step, indicator=None):
        pass


class Scalar(Metric):
    def __init__(self, writer: SummaryWriter, name):
        super(Scalar, self).__init__(writer, name)

    def update(self, data, step, indicator=None):
        if indicator is not None:
            tag = f"{self.name}/{indicator}"
        else:
            tag = self.name
        self.writer.add_scalar(tag, data, step)


class Image(Metric):
    def __init__(self, writer: SummaryWriter, name: str):
        super(Image, self).__init__(writer, name)

    def update(self, image_data, step, indicator=None, flip_height=True):
        if indicator is not None:
            tag = f"{self.name}/{indicator}"
        else:
            tag = self.name
        if type(image_data) == Tensor:
            height_channel = 2 if len(image_data.shape) == 4 else 1
            if flip_height:
                image_data = torch.flip(image_data, [height_channel])
            if len(image_data.shape) == 4:
                self.writer.add_images(tag, image_data, step)
            elif len(image_data.shape) == 4:
                self.writer.add_image(tag, image_data, step)
        else:
            self.writer.add_figure(tag, image_data, step)


class Audio(Metric):
    def __init__(self, writer: SummaryWriter, name: str):
        super(Audio, self).__init__(writer, name)

    def update(self, data, step, indicator=None, sr=44100):
        if indicator is not None:
            tag = f"{self.name}/{indicator}"
        else:
            tag = self.name
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                self.writer.add_audio(tag + f"-{i}", data[i, :], step, sample_rate=sr)

        else:
            self.writer.add_audio(tag, data, step, sample_rate=sr)


class SpectralAudio(Audio):
    def __init__(self, writer: SummaryWriter, gabor_inverter: GaborInverter):
        super(Audio, self).__init__(writer, "Inverted Spectrogram Audio")
        self.inverter = gabor_inverter

    def update(self, data, step, indicator=None, sr=44100):
        batch_size = data.shape[0]
        for i in range(batch_size):
            reconstructed_audio = self.inverter(data[i].detach().cpu())

            super().update(reconstructed_audio, step, f"{indicator}-{i}", sr)


class Loss(Scalar):
    def __init__(self, writer: SummaryWriter):
        super(Loss, self).__init__(writer, "Loss")

    def update(self, data, step, indicator=None):
        super().update(data, step, indicator)


class GradientPenalty(Scalar):
    def __init__(self, writer: SummaryWriter):
        super(GradientPenalty, self).__init__(writer, "Gradient Penalty")

    def update(self, data, step, indicator=None):
        super().update(data, step, indicator)


class LearningRate(Scalar):
    def __init__(self, writer: SummaryWriter):
        super(LearningRate, self).__init__(writer, "Learning Rate")

    def update(self, data, step, indicator=None):
        super().update(data, step, indicator)


class ImageSample(Image):
    def __init__(self, writer: SummaryWriter):
        super(ImageSample, self).__init__(writer, "Sample")

    def update(self, data, step, indicator=None, flip_height=True):
        super().update(data, step, indicator)


class SpectrogramSample(Image):
    def __init__(self, writer: SummaryWriter, gabor_inverter: GaborInverter = None, data_parameters: DataParameters = None):
        super(SpectrogramSample, self).__init__(writer, "Spectrogram Sample")
        self.inverter = gabor_inverter
        self.data_parameters = data_parameters

    def _update_with_decompressed(self, spectrogram, step, indicator, suffix=None):
        decompressed = self.inverter.decompressor(spectrogram.transpose(1, 2).unsqueeze(0).detach().cpu()).transpose(1, 2)
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
        if len(decompressed.shape) == 3:
            decompressed = decompressed[0]

        figure, axs = plt.subplots(dpi=300, nrows=1, ncols=2, sharex=True)
        axs[0].imshow(spectrogram.detach().cpu().numpy(), origin="lower", interpolation=None)
        axs[0].set_title("Frequency Compressed")

        axs[1].imshow(decompressed.detach().cpu().numpy(), origin="lower", interpolation=None)
        axs[1].set_title("Decompressed")
        if suffix is not None:
            indicator = f"{indicator} {suffix}"
        super().update(figure, step, indicator)

    def _update_single(self, spectrogram, step, indicator, suffix=None):
        figure, ax = plt.subplots(dpi=300)
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
        ax.imshow(spectrogram.detach().cpu().numpy(), interpolation=None, origin='lower')
        if suffix is not None:
            indicator = f"{indicator} {suffix}"
        super().update(figure, step, indicator)

    def _update(self, spectrogram, step, indicator, suffix=None):
        if self.data_parameters is not None and self.inverter is not None and self.data_parameters.frequency_compression.active:
            self._update_with_decompressed(spectrogram, step, indicator, suffix)
        else:
            self._update_single(spectrogram, step, indicator, suffix)

    def update(self, data, step, indicator=None, flip_height=True):
        spectrogram_count = data.shape[0]
        if spectrogram_count == 1:
            self._update(data, step, indicator)
        else:
            for i in range(spectrogram_count):
                self._update(data[i], step, indicator, suffix=i)


class SampleHistogram(Image):
    def __init__(self, writer: SummaryWriter):
        super(SampleHistogram, self).__init__(writer, "Sample Histogram")

    def update(self, image_data, step, indicator=None, flip_height=False):
        spectrogram_count = image_data.shape[0]
        figure, ax = plt.subplots(dpi=300, nrows=1, ncols=spectrogram_count, sharey=True)
        for idx in range(spectrogram_count):
            spectrogram = image_data[idx]
            if len(spectrogram.shape) == 3:
                spectrogram = spectrogram[0]

            ax[idx].hist(torch.flatten(spectrogram.detach().cpu()).numpy())
            ax[idx].tick_params(axis='x', labelsize=4)

        super().update(figure, step, indicator)