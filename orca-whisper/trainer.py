import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable, grad
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging import Logger
from utils.noise import NoiseGenerator
from models.orcawhisper import OrcaGANGenerator, OrcaGANDiscriminator
from utils.parameter import TrainingParameters, LatentSpaceParameters, GeneralParameters
from utils.checkpoint import restore_checkpoint, save_checkpoint
from utils.model import weights_init_xavier

torch.cuda.empty_cache()


class Trainer:
    def __init__(
            self,
            data_set,
            directories,
            noise_generator: NoiseGenerator,
            logger: Logger,
            training_parameters: TrainingParameters,
            general_parameters: GeneralParameters,
            latent_space_parameters: LatentSpaceParameters,
            metrics=None,

    ):
        self.checkpoint_dir, self.log_dir, self.summary_dir, self.landmark_dir = directories
        self.log = logger
        self.noise_generator = noise_generator
        self.training_parameters = training_parameters
        self.general_parameters = general_parameters
        self.critic_iterations = training_parameters.d_iterations
        self.generator_iterations = training_parameters.iterations

        self.generator = OrcaGANGenerator(
            latent_dimension=latent_space_parameters.dimension).apply(weights_init_xavier)
        self.discriminator = OrcaGANDiscriminator().apply(weights_init_xavier)

        self.batch_size = training_parameters.batch_size
        self.latent_dim = latent_space_parameters.dimension
        self.gradient_penalty_lambda = training_parameters.gradient_penalty_lambda
        self.metrics = metrics

        self.log.debug("Initialize Data")
        self.data_set = data_set
        # initialize dataloader and a summery writer
        self.data_loader = DataLoader(self.data_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.use_cuda = False
        else:
            self.device = torch.device("cuda:0")
            self.use_cuda = True

        checkpoint_data, ckp_number = restore_checkpoint(self.checkpoint_dir,
                                                         generator=self.generator,
                                                         discriminator=self.discriminator,
                                                         device=self.device)
        if ckp_number != -1:
            self.log.info(f'Restoring from checkpoint {ckp_number}')
        else:
            self.log.info(f"No checkpoint found. Starting from scratch")

        self.iteration = checkpoint_data['iteration']
        self.g_losses = checkpoint_data['g_losses']
        self.d_losses = checkpoint_data['d_losses']
        self.gradient_penalties = []

        self.beta1 = training_parameters.beta_1
        self.beta2 = training_parameters.beta_2
        self.lr = training_parameters.learning_rate
        self.D_opt = Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.G_opt = Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        self.best_dict = checkpoint_data["best"]
        self.log.info(OrderedDict({
            "Generator": self.generator,
            "Discriminator": self.discriminator
        }))

    def get_gradient_penalty(self, real_batch, fake_batch):
        """
          Calculates the WGAN-GP Gradient Penalty
        """
        alpha = torch.rand(self.batch_size, 1, 1, 1)
        alpha = alpha.expand(real_batch.size()).to(self.device)

        interpolates = alpha * real_batch + ((1 - alpha) * fake_batch)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discriminator(interpolates)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(self.batch_size, -1)

        gradients_norm = (gradients + 1e-12).norm(2, dim=1)
        gradients_norm = gradients_norm - 1
        gradients_norm = gradients_norm ** 2
        gradients_norm = gradients_norm.mean()
        gradient_penalty = self.gradient_penalty_lambda * gradients_norm

        return gradient_penalty

    def report_metrics(self, g_loss, d_loss, real_samples, fake_samples, gradient_penalty, iteration, best_dict=None):
        self.metrics["Discriminator"]["Loss"].update(d_loss, iteration, "Discriminator")
        self.metrics["Generator"]["Loss"].update(g_loss, iteration, "Generator")
        self.metrics["Training"]["GradientPenalty"].update(gradient_penalty, iteration)
        self.metrics["Training"]["LearningRate"]["Discriminator"].update(self.G_opt.param_groups[0]["lr"], iteration,
                                                                         "Discriminator")
        self.metrics["Training"]["LearningRate"]["Generator"].update(self.G_opt.param_groups[0]["lr"], iteration,
                                                                     "Generator")

        self.metrics["Samples"]["Real"].update(real_samples, iteration, "Real")
        self.metrics["Samples"]["Fake"].update(fake_samples, iteration, "Fake")
        self.metrics["Samples"]["RealAudio"].update(real_samples, iteration, "RealAudio")
        self.metrics["Samples"]["FakeAudio"].update(fake_samples, iteration, "FakeAudio")
        self.log.info(
            f"step: {iteration}|Loss Discriminator: {d_loss}|Loss Generator: {g_loss}|Gradient Penalty: {gradient_penalty}|")

    def save(self, iteration, final=False):
        if not final:
            if iteration % self.training_parameters.checkpointing.iterations_per_checkpoint == 0 and iteration >= 1:
                save_checkpoint(generator=self.generator,
                                discriminator=self.discriminator,
                                checkpoint_directory=self.checkpoint_dir,
                                g_losses=self.g_losses,
                                d_losses=self.d_losses,
                                checkpoint_number=iteration,
                                best_dict=self.best_dict
                                )

            if iteration % self.training_parameters.checkpointing.iterations_per_landmark == 0 and iteration >= 1:
                save_checkpoint(generator=self.generator,
                                discriminator=self.discriminator,
                                checkpoint_directory=self.landmark_dir,
                                g_losses=self.g_losses,
                                d_losses=self.d_losses,
                                checkpoint_number=iteration,
                                best_dict=self.best_dict
                                )
        else:
            save_checkpoint(generator=self.generator,
                            discriminator=self.discriminator,
                            checkpoint_directory=self.training_parameters.base_training_directory,
                            checkpoint_number=iteration,
                            final=True,
                            class_name=self.general_parameters.class_name
                            )

    def fit(self):
        self.log.debug("Start Training")
        iteration = self.iteration

        try:
            for iteration_g in tqdm(range(self.iteration, self.generator_iterations)):
                _d_losses = []
                gps = []
                real_samples = None
                fake_samples = None
                for iteration_d in range(self.critic_iterations):
                    """
                    Train the Discriminator for several iterations
                    """
                    self.D_opt.zero_grad()
                    data = next(iter(self.data_loader)).to(self.device)
                    noise = self.noise_generator.sample().to(self.device).double()

                    G_z = self.generator(noise)

                    if real_samples is None:
                        real_samples = data
                        fake_samples = G_z

                    D_x = self.discriminator(data).mean()
                    D_G_z = self.discriminator(G_z).mean()

                    gradient_penalty = self.get_gradient_penalty(data, G_z)

                    D_loss = D_G_z - D_x + gradient_penalty
                    _d_losses.append(D_loss.item())
                    D_loss.backward()

                    gps.append(gradient_penalty.item())
                    if np.abs(D_loss.item()) < self.best_dict["discriminator"]["loss"]:
                        self.best_dict["discriminator"]["model_state_dict"] = self.discriminator.state_dict()
                        self.best_dict["discriminator"]["iteration"] = iteration_g
                        self.best_dict["discriminator"]["loss"] = np.abs(D_loss.item())

                    self.D_opt.step()

                """
                Train the Generator
                """

                self.G_opt.zero_grad()
                noise = self.noise_generator.sample().to(self.device).double()
                G_z = self.generator(noise)

                D_G_z = self.discriminator(G_z).mean()
                G_loss = -D_G_z
                G_loss.backward()

                d_loss = np.mean(_d_losses)

                self.g_losses.append(G_loss.item())
                self.d_losses.append(d_loss)

                self.G_opt.step()

                iteration = iteration_g

                if np.abs(G_loss.item()) < self.best_dict["generator"]["loss"]:
                    self.best_dict["generator"]["model_state_dict"] = self.generator.state_dict()
                    self.best_dict["generator"]["iteration"] = iteration_g
                    self.best_dict["generator"]["loss"] = np.abs(G_loss.item())
                    self.best_dict["generator"]["samples"] = G_z[0:3]

                if iteration % self.training_parameters.metric_iterations == 0:
                    self.report_metrics(g_loss=G_loss,
                                        d_loss=d_loss,
                                        real_samples=real_samples,
                                        fake_samples=fake_samples,
                                        gradient_penalty=np.mean(gps),
                                        iteration=iteration,
                                        best_dict=self.best_dict)

                self.save(iteration_g)

            self.save(iteration, final=True)

        except (KeyboardInterrupt, SystemExit):
            self.save(iteration, final=True)
            sys.exit(0)

        return self.generator, self.discriminator
