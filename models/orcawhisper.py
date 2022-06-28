import torch
import torch.nn as nn


class OrcaGANGenerator(nn.Module):
    """
    Here, the network structures of ORCA-WHISPER is implemented
    """
    def __init__(self, model_size=256, kernel_size=(12, 3), stride=(2, 2), padding=(5, 1), output_padding=(0, 1),
                 latent_dimension=100, scale=8):
        super(OrcaGANGenerator, self).__init__()

        self.project = nn.Sequential(
            Project(latent_dimension=latent_dimension,
                    scale=scale,
                    dimension=model_size)
        )

        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32 * scale,
                out_channels=32 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32 * scale,
                out_channels=16 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16 * scale,
                out_channels=8 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8 * scale,
                out_channels=4 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4 * scale,
                out_channels=2 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2 * scale,
                out_channels=scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=scale,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        output = self.project(input_tensor)
        output = self.deconv0(output)
        output = self.deconv1(output)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        output = self.deconv6(output)

        return output


class Project(nn.Module):
    def __init__(self, latent_dimension=100, dimension=256, dimension_multiplier=4, scale=32):
        super(Project, self).__init__()
        self.dimension = dimension
        self.dimension_multiplier = dimension_multiplier
        self.scale = scale
        self.fc = nn.Linear(latent_dimension, dimension * scale)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        output = self.fc(input_tensor)
        output = output.reshape([batch_size, self.scale * 32,  self.dimension_multiplier,
                                 self.dimension_multiplier // 2])
        return output


class OrcaGANDiscriminator(nn.Module):
    def __init__(self, leaky_relu_alpha=0.2, kernel_size=(12, 3), stride=(2, 2), padding=(5, 1), scale=8, final_size=256):
        super(OrcaGANDiscriminator, self).__init__()
        self.scale = scale
        self.final_size = final_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_alpha,
                inplace=True
            )
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=scale,
                out_channels=2 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_alpha,
                inplace=True
            )
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * scale,
                out_channels=4 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_alpha,
                inplace=True
            )
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * scale,
                out_channels=8 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_alpha,
                inplace=True
            )
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=8 * scale,
                out_channels=16 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_alpha,
                inplace=True
            )
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16 * scale,
                out_channels=32 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_alpha,
                inplace=True
            )
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=32 * scale,
                out_channels=32 * scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(
                negative_slope=leaky_relu_alpha,
                inplace=True
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=self.final_size * scale,
                out_features=1)
        )

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        output = self.conv1(input_tensor)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = output.reshape(batch_size, self.final_size * self.scale)
        output = self.fc(output)

        return output


if __name__ == '__main__':
    batch = 8
    latent_dim = 100
    ckp_file = "/media/alex/s1/experiments/ORCA-WHISPER/CHRISTIAN/PARAKEET/contact/CHECKPOINTS/ckp-10280.ckp"
    data = torch.load(ckp_file)
    noise = torch.normal(mean=0, std=1, size=(batch, latent_dim))
    generator = OrcaGANGenerator(latent_dimension=latent_dim)
    generator.load_state_dict(data["generator_state_dict"])
    discriminator = OrcaGANDiscriminator()
    discriminator.load_state_dict(data["discriminator_state_dict"])
    G_z = generator(noise)
    D_G_z = discriminator(G_z)
    print(D_G_z.shape)
