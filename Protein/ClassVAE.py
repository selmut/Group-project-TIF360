import torch
import torchvision.models as models


class VariationalEncoder(torch.nn.Module):
    def __init__(self, latent_dims, img_dim, img_channels=3):
        super(VariationalEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(img_channels, 64, kernel_size=7, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=7, stride=2)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.conv5 = torch.nn.Conv2d(512, 3, kernel_size=3, stride=2)

        self.linear1 = torch.nn.Linear(169, latent_dims)
        self.linear2 = torch.nn.Linear(169, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.mu = 0
        self.sigma = 1
        self.kl = 0

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        self.mu = self.linear1(x)
        self.sigma = torch.exp(self.linear2(x))

        z = self.mu + self.sigma*self.N.sample(self.mu.shape)

        self.kl = (self.sigma**2 + self.mu**2 - torch.log(self.sigma) - 1/2).sum()
        return z


class Decoder(torch.nn.Module):
    def __init__(self, latent_dims, img_dim):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dims, 169)
        self.upsample1 = torch.nn.UpsamplingBilinear2d(size=(64, 64))
        self.upsample2 = torch.nn.UpsamplingBilinear2d(size=(128, 128))
        self.upsample3 = torch.nn.UpsamplingBilinear2d(size=(256, 256))
        self.upsample4 = torch.nn.UpsamplingBilinear2d(size=(512, 512))

    def forward(self, z):
        z = self.linear1(z)
        z = z.reshape((-1, 3, 13, 13))
        z = torch.nn.functional.relu(self.upsample1(z))
        z = torch.nn.functional.relu(self.upsample2(z))
        z = torch.nn.functional.relu(self.upsample3(z))
        z = torch.sigmoid(self.upsample4(z))
        return z


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dims, img_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, img_dim)
        self.decoder = Decoder(latent_dims, img_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



