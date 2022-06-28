import torch
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.naming import OrcaSpotNameGenerator
from utils.parameter import GaussianLatentSpaceParameters, Parameters

from utils.checkpoint import get_checkpoints
from utils.system import make_folders


from utils.noise import GaussianNoiseGenerator
from models.orcawhisper import OrcaGANGenerator
import numpy as np
import soundfile as sf
import utils.transforms as T


class GANSampler:
    def __init__(self,
                 output_base,
                 name_generator: OrcaSpotNameGenerator,
                 checkpoint_path,
                 noise_generator,
                 device,
                 inverter,
                 n_fft=4096,
                 hop_length=256,
                 power_to_mag=True,
                 class_name="N1",
                 load_best_generator=False,
                 latent_dimension=100,
                 f_min=0,
                 f_max=10000,
                 sr=44100
                 ):
        self.output_base = output_base
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.checkpoint_path = checkpoint_path
        self.load_best_generator = load_best_generator
        self.latent_dimension = latent_dimension

        self.power_to_mag = power_to_mag
        self.noise_generator = noise_generator
        self.device = device
        self.class_name = class_name
        self.inverter = inverter

        self.img_output = os.path.join(output_base, "IMG")
        self.audio_output = os.path.join(output_base, "WAV")
        self.name_generator = name_generator
        self.generator = self.load_generator()
        self.f_min = f_min
        self.f_max = f_max
        self.sr = sr

        make_folders([self.img_output, self.audio_output])

    def load_generator(self):
        print("Fetching Generator")
        data = torch.load(self.checkpoint_path)
        state_dict = data["generator_state_dict"]
        if self.load_best_generator and "best" in data:
            state_dict = data["best"]["generator"]["model_state_dict"]

        gen = OrcaGANGenerator(latent_dimension=self.latent_dimension)
        gen.load_state_dict(state_dict)
        gen.to(self.device)
        return gen


    def write_audio(self, X, name):
        if len(X.shape) == 4:
            X = X[0]
        audio = self.inverter(X)

        if len(audio.shape) == 2:
            audio = audio[0, :]
        sf.write(os.path.join(self.audio_output, name), audio, samplerate=self.sr)

    def write_spectrogram(self, spectrogram, name, transpose=False):
        spectrogram = spectrogram.squeeze().detach().cpu().numpy()
        if transpose:
            spectrogram = spectrogram.T

        figure, ax = plt.subplots(dpi=300)
        plt.imshow(spectrogram, origin="lower", interpolation=None)
        plt.savefig(os.path.join(self.img_output, name), bbox_inches="tight")
        plt.close(figure)

    def plot_hist(self, spectrogram):
        sample = torch.flatten(spectrogram).detach().cpu().numpy()
        _mean = np.round(np.mean(sample), 2)
        _min = np.round(np.min(sample), 2)
        _max = np.round(np.max(sample), 2)
        plt.hist(sample)
        plt.title(f"Mean: {_mean} -- Min: {_min} -- Max:{_max}")
        plt.show()

    def __call__(self, sample_count):
        for _ in tqdm(range(sample_count)):
            z = self.noise_generator.sample().to(self.device)
            G_z = self.generator(z)

            G_z.clamp_(min=0, max=1)

            output_name = self.name_generator.generate(self.class_name)
            self.write_spectrogram(G_z, output_name)
            self.write_audio(G_z, output_name.replace(".png", ".wav"))


class GANExperimentSampler:
    def __init__(self,
                 experiment_path,
                 name_generator,
                 device,
                 inverter: T.GaborInverter,
                 sample_count=10,
                 landmark_count=5,
                 checkpoint_count=10,
                 class_name="ORCA",
                 latent_dimension=100,
                 batch_size=1,
                 f_min=0,
                 f_max=10000
                 ):
        self.noise_generator = GaussianNoiseGenerator(parameters=GaussianLatentSpaceParameters(),
                                                      batch_size=batch_size,
                                                      latent_dimension_size=latent_dimension)
        self.name_generator = name_generator
        self.inverter = inverter
        self.sample_count = sample_count
        self.class_name = class_name
        self.device = device
        self.checkpoint_count = checkpoint_count
        self.landmark_count = landmark_count
        self.checkpoint_directory = os.path.join(experiment_path, "CHECKPOINTS")
        self.landmark_directory = os.path.join(experiment_path, "LANDMARKS")
        self.selected_directory = os.path.join(experiment_path, "SELECTED")
        self.base_output_directory = os.path.join(experiment_path, "SAMPLES")
        self.f_min = f_min
        self.f_max = f_max

    def _sample(self, checkpoint_path):
        GANSampler(
            name_generator=self.name_generator,
            class_name=self.class_name,
            checkpoint_path=checkpoint_path,
            noise_generator=self.noise_generator,
            device=self.device,
            f_min=self.f_min,
            f_max=self.f_max,
            inverter=self.inverter,
            output_base=os.path.join(self.base_output_directory,
                                     f"{self.class_name}-{os.path.basename(checkpoint_path).replace('.ckp', '')}")
        )(self.sample_count)

    def __call__(self):
        checkpoints = get_checkpoints(self.checkpoint_directory)[-self.checkpoint_count:]
        landmarks = get_checkpoints(self.landmark_directory)[-self.landmark_count:]
        selected = [] if not os.path.isdir(self.selected_directory) else get_checkpoints(self.selected_directory)
        for checkpoint in checkpoints:
            self._sample(checkpoint)
        for landmark in landmarks:
            self._sample(landmark)
        for s in selected:
            self._sample(s)


if __name__ == '__main__':
    count = 100
    use_cuda = False
    batch_size = 1
    latent_dimension = 100

    if torch.cuda.is_available() and use_cuda:
        d = torch.device("cuda:0")
    else:
        d = torch.device("cpu")

    experiment = f"/media/alex/s1/experiments/BIO-GAN/MONK-PARAKEET/other-compressed-high-time-res-dynamic-hop"
    experiment_param_file = os.path.join(experiment, "parameters.json")
    experiment_params = Parameters(experiment_param_file)

    g_inverter = T.GaborInverter(experiment_params.data, device=d)
    name_gen = OrcaSpotNameGenerator()
    output_dir = os.path.join(experiment, "SAMPLES")
    class_name = "other"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    GANExperimentSampler(
        experiment_path=experiment,
        inverter=g_inverter,
        name_generator=name_gen,
        device=d,
        class_name=class_name,
        sample_count=1000,
        landmark_count=3,
        checkpoint_count=10,
        f_min=0,
        f_max=10000
    )()

