import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np
import torch#; torch.manual_seed(0)


def plot_reconstructed(autoencoder, filename, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    plt.figure()
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]])
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1], cmap='Greys')
    plt.savefig('img/'+filename)
    plt.close()


def plot_latent(autoencoder, data, num_batches=100):
    plt.figure()
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.savefig('img/vae.png')
    plt.close()

