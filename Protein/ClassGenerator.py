import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import io
import PIL

import plots
from ClassVAE import VariationalAutoencoder


class DataGenerator:
    def __init__(self, dataset, latent_dims):
        self.dataset = dataset
        self.labels = set(dataset.targets.numpy())

        self.latent_dims = latent_dims
        self.vae = VariationalAutoencoder(self.latent_dims)
        self.encoder = self.vae.encoder
        self.decoder = self.vae.decoder

    def get_data_by_label(self, label):
        indices = [idx for idx, target in enumerate(self.dataset.targets) if target in [label]]
        dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataset, indices), batch_size=128, drop_last=True)
        return dataloader

    def train_network(self, dataloader, epochs=20):
        opt = torch.optim.Adam(self.vae.parameters())
        for epoch in range(epochs):
            for x, y in dataloader:
                opt.zero_grad()
                x_hat = self.vae(x)
                loss = ((x - x_hat) ** 2).sum() + self.vae.encoder.kl
                loss.backward()
                opt.step()

            print(f'Epoch nr. {epoch + 1:02d}/{epochs} --- Current training loss: {loss:.4f}')

        torch.save(self.vae, 'models/variational_autoencoder.pt')

    def generate_new_dataset(self, output_size=100, img_dim=28):

        labels_array = np.zeros(output_size*len(self.labels))
        img_names = []

        for idx, label in enumerate(self.labels):
            print(f'\nGenerating data for label \'{label}\'')
            dataloader = self.get_data_by_label(label)

            # print('\nTraining variational autoencoder...\n')
            self.train_network(dataloader)

            U = torch.distributions.Uniform(-2, 2)
            for i in range(output_size):
                point = U.sample(sample_shape=[self.latent_dims])
                z = torch.Tensor(point)
                x_hat = self.vae.decoder(z)
                x_hat = x_hat.reshape((img_dim, img_dim)).detach().numpy()

                x_hat_scaled = (((x_hat[:, :] - x_hat[:, :].min()) / (x_hat[:, :].max() - x_hat[:, :].min())) * 255.9).astype(np.uint8)
                img = Image.fromarray(x_hat_scaled)

                img.save(f'data/GeneratedMNIST/img/{label}_{i}.png')

                labels_array[output_size*idx+i] = label
                img_names.append(f'{label}_{i}.png')

            if self.latent_dims == 2:
                plots.plot_reconstructed(self.vae, f'reconstructed_{label}.png', r0=(-3, 3), r1=(-3, 3))

        labels_df = pd.DataFrame(labels_array, columns=['label'], index=None)
        labels_df['image_name'] = img_names
        labels_df.to_csv('data/GeneratedMNIST/labels.csv', index=False)


