import torch#; torch.manual_seed(0)
from aeClass import Decoder


class VariationalEncoder(torch.nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, latent_dims)
        self.linear3 = torch.nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N_fit = torch.distributions.Normal(0, 1)
        self.mu = 0
        self.sigma = 1
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.linear1(x))
        self.mu = self.linear2(x)
        self.sigma = torch.exp(self.linear3(x))
        z = self.mu + self.sigma*self.N.sample(self.mu.shape)
        self.kl = (self.sigma**2 + self.mu**2 - torch.log(self.sigma) - 1/2).sum()
        self.N_fit = torch.distributions.Normal(self.mu[-1], self.sigma[-1])
        return z


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
