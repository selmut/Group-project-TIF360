import torch#; torch.manual_seed(0)
import torch.nn.functional as F

#img_size = 28*28*1
class VariationalEncoder(torch.nn.Module):
    def __init__(self, latent_dims, feature_dim = 32*20*20, z_dim = 256):
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
        self.N_fit = torch.distributions.Normal(self.mu[-1], self.sigma[-1])
        return z

class Decoder(torch.nn.Module):
    def __init__(self, latent_dims, z_dim = 256, feature_dim = 32*20*20):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dims, 512)
        self.linear2 = torch.nn.Linear(512, 784)

    def forward(self, z):
        z = torch.nn.functional.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))

        return z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



