import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataset import GaborCache, GaborDataset
from utils.transforms import GaborInverter
from utils.logging import Logger
from utils.metrics import Loss, GradientPenalty, LearningRate, SpectrogramSample, SpectralAudio
from utils.noise import NoiseGenerator
from utils.parameter import Parameters
from trainer import Trainer
import shutil

from utils.system import make_folders

torch.cuda.empty_cache()


if __name__ == "__main__":
    parameter_file = os.path.join(os.getcwd(), "parameters.json")
    parameters = Parameters(parameter_file)

    _base_training_directory = parameters.training.base_training_directory
    summary_directory = os.path.join(_base_training_directory, "SUMMARY")
    log_directory = os.path.join(_base_training_directory, "LOG")
    checkpoint_directory = os.path.join(_base_training_directory, "CHECKPOINTS")
    landmark_directory = os.path.join(_base_training_directory, "LANDMARKS")
    make_folders([_base_training_directory, summary_directory, log_directory, checkpoint_directory, landmark_directory])
    shutil.copy(parameter_file, _base_training_directory)

    log = Logger("Train", debug=True, log_dir=log_directory)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        log.info("Not using GPU for training")
    else:
        device = torch.device("cuda:0")
        log.info("Using GPU for training")

    writer = SummaryWriter(log_dir=summary_directory, flush_secs=10)

    noise_generator = NoiseGenerator(parameters.latent_space, parameters.training)
    g_cache = GaborCache(parameters.cache)

    g_inverter = GaborInverter(parameters.data,
                               device=device)
    dataset = GaborDataset(
        parameters=parameters.data,
        cache=g_cache,
        device=device,
        log=log
    )

    log.info(f"Looking for data in {parameters.data.data_directory}")
    log.info(f"Found {len(dataset)} files for training")
    log.info(f"Training Parameters: ")
    log.info(json.dumps(parameters.training.json_data))

    metric_dict = {
        "Discriminator": {
            "Loss": Loss(writer)
        },
        "Generator": {
            "Loss": Loss(writer)
        },
        "Samples": {
            "Real": SpectrogramSample(writer, gabor_inverter=g_inverter, data_parameters=parameters.data),
            "Fake": SpectrogramSample(writer, gabor_inverter=g_inverter, data_parameters=parameters.data),
            "RealAudio": SpectralAudio(writer, gabor_inverter=g_inverter),
            "FakeAudio": SpectralAudio(writer, gabor_inverter=g_inverter),

        },
        "Training": {
            "GradientPenalty": GradientPenalty(writer),
            "LearningRate": {
                "Discriminator": LearningRate(writer),
                "Generator": LearningRate(writer),

            }
        }
    }

    trainer = Trainer(
        data_set=dataset,
        directories=[checkpoint_directory, log_directory, summary_directory, landmark_directory],
        logger=log,
        noise_generator=noise_generator,
        training_parameters=parameters.training,
        general_parameters=parameters.general,
        latent_space_parameters=parameters.latent_space,
        metrics=metric_dict
    )

    generator, discriminator = trainer.fit()
