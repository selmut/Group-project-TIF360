import torch#; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import plots
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import seaborn as sns

from vaeClass import Network

latent_dims = 2
net = Network(latent_dims)


print('Loading data...')
'''data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False).__getitem__(0),
        batch_size=128, shuffle=True)'''

dataset = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False)

indices = [idx for idx, target in enumerate(dataset.targets) if target in [9]]
data = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=128, drop_last=True)


print('\nTraining variational autoencoder...\n')
net.train_vae(data)
torch.save(net.vae, 'models/variational_autoencoder.pt')

U = torch.distributions.Uniform(-3, 3)

'''for i in range(100):
    # point = vae.encoder.mu[-1] + vae.encoder.sigma[-1]*N.sample([latent_dims])
    point = U.sample(sample_shape=[latent_dims])
    z = torch.Tensor(point)
    x_hat = vae.decoder(z)
    x_hat = x_hat.reshape((28, 28)).detach().numpy()
    plt.figure()
    plt.imshow(x_hat[:, :], cmap='Greys')
    plt.title(f'Sampled point = {[np.round(x, 4) for x in list(point.detach().numpy())]}')
    plt.savefig(f'img/digits/{i}.png')
    plt.close()'''

plots.plot_latent(net.vae, data)
plots.plot_reconstructed(net.vae, r0=(-3, 3), r1=(-3, 3))




