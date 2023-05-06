import torch#; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import plots
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from vaeClass import VariationalAutoencoder

# Reference: https://avandekleut.github.io/vae/


def train_vae(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()

        print(f'Epoch nr. {epoch + 1}/{epochs} --- Current training loss: {loss:.4f}')
    return autoencoder


latent_dims = 2
vae = VariationalAutoencoder(latent_dims)  # GPU

print('Loading data...')
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False),
        batch_size=128, shuffle=True)

print('\nTraining variational autoencoder...\n')
vae = train_vae(vae, data)


plots.plot_latent(vae, data)
plots.plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))

