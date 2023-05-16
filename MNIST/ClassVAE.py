import torch  # ; torch.manual_seed(0)
import torch.nn.functional as F


# img_size = 28*28*1
class VariationalEncoder(torch.nn.Module):
    def __init__(self, latent_dims, feature_dim=32 * 20 * 20, z_dim=256):
        super(VariationalEncoder, self).__init__()
        self.eConv1 = torch.nn.Conv2d(1, 16, 5)
        self.eConv2 = torch.nn.Conv2d(16, 32, 5)
        self.eLinear1 = torch.nn.Linear(feature_dim, z_dim)
        self.eLinear2 = torch.nn.Linear(feature_dim, z_dim)

        # Selma --------------------------------------------------------------------------------------------------------
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, latent_dims)
        self.linear3 = torch.nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N_fit = torch.distributions.Normal(0, 1)
        self.mu = 0
        self.sigma = 1
        self.kl = 0

    def forward(self, x):
        x = self.eConv1(x)
        x = F.relu(x)
        x = self.eConv2(x)
        x = F.relu(x)
        x = x.view(-1, 32 * 20 * 20)  # Ska va samma som feature_dim i _init_
        mu = self.eLinear1(x)
        logVar = self.eLinear2(x)

        std = torch.exp(logVar / 2)
        eps = torch.rand_like(std)

        z = mu + eps * std

        # Selma -------------------------------------------------------------------------------
        '''x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.linear1(x))
        self.mu = self.linear2(x)
        self.sigma = torch.exp(self.linear3(x))

        self.kl = (self.sigma**2 + self.mu**2 - torch.log(self.sigma) - 1/2).sum()
        self.N_fit = torch.distributions.Normal(self.mu[-1], self.sigma[-1])'''
        return z


class Decoder(torch.nn.Module):
    def __init__(self, latent_dims, z_dim=256, feature_dim=32 * 28 * 28):
        super(Decoder, self).__init__()
        self.dLinear = torch.nn.Linear(z_dim, feature_dim)
        self.dConv1 = torch.nn.ConvTranspose2d(32, 16, 5)
        self.dConv2 = torch.nn.ConvTranspose2d(16, 1, 5)

        # Selma --------------------------------------------------------------------------------------------------------
        self.linear1 = torch.nn.Linear(latent_dims, 512)
        self.linear2 = torch.nn.Linear(512, 784)

    def forward(self, z):
        z = self.dLinear(z)
        z = F.relu(z)
        z = z.view(-1, 32, 28, 28)
        z = self.dConv1(z)
        z = F.relu(z)
        z = self.dConv2(z)
        z = torch.sigmoid(z)

        '''z = torch.nn.functional.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))'''

        return z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
