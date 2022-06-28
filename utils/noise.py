import torch
from torch import Tensor

from utils.parameter import LatentSpaceParameters, GaussianLatentSpaceParameters, \
    UniformLatentSpaceParameters, TrainingParameters


class NoiseGenerator:
    def __init__(self, latent_space_parameters: LatentSpaceParameters, training_parameters: TrainingParameters):
        if latent_space_parameters.latent_space_generator == "gaussian":
            self.generator = GaussianNoiseGenerator(
                latent_space_parameters.data,
                training_parameters.batch_size,
                latent_space_parameters.dimension)
        else:
            self.generator = UniformNoiseGenerator(
                latent_space_parameters.data,
                training_parameters.batch_size,
                latent_space_parameters.dimension
            )

    def sample(self) -> Tensor:
        return self.generator.sample()


class GaussianNoiseGenerator:
    def __init__(self, parameters: GaussianLatentSpaceParameters, batch_size, latent_dimension_size):
        self.mean = parameters.mean
        self.std = parameters.std
        self.batch_size = batch_size
        self.latent_dimension_size = latent_dimension_size

    def sample(self):
        return torch.normal(mean=0, std=1, size=(self.batch_size, self.latent_dimension_size))


class UniformNoiseGenerator:
    def __init__(self, parameters: UniformLatentSpaceParameters, batch_size, latent_dimension_size):

        self.batch_size = batch_size
        self.latent_dimension_size = latent_dimension_size

    def sample(self) -> Tensor:
        return torch.rand(self.batch_size, self.latent_dimension_size)


if __name__ == '__main__':
    data_dir = "/media/alex/Big Storage 2/ORCHIVE"
