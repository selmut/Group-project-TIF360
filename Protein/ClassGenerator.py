import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import io
import torchvision.transforms as T
import PIL
import plots
from ClassConvVAE import VariationalAutoencoder
from ClassResnetVAE import ResNetVAE


class DataGenerator:
    def __init__(self, dataset, latent_dims, in_channels):
        self.dataset = dataset
        self.labels = set(dataset.targets)

        self.latent_dims = latent_dims
        self.vae = VariationalAutoencoder(self.latent_dims, in_channels)
        self.encoder = self.vae.encoder
        self.decoder = self.vae.decoder

    def get_data_by_label(self, label):
        indices = [idx for idx, target in enumerate(self.dataset.targets) if target in [label]]
        dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataset, indices), batch_size=128,
                                                 drop_last=True, shuffle=True)
        return dataloader

    def train_network(self, dataloader, epochs=20):
        n = 0
        for epoch in range(epochs):
            losses = np.zeros(len(dataloader))
            lrs = np.linspace(0.001, 0.0001, num=epochs)
            for idx, (x, y) in enumerate(dataloader):
                opt = torch.optim.Adam(self.vae.parameters(), amsgrad=True, lr=lrs[n])
                opt.zero_grad()
                x_hat = self.vae(x)
                loss = ((x - x_hat) ** 2).sum() + self.vae.encoder.kl
                loss.backward()
                opt.step()
                losses[idx] = loss
                print(f'Batch nr. {idx + 1}/{len(dataloader)} --- Current batch loss: {loss:.4f}')
            n += 1
            print(f'\nEpoch nr. {epoch + 1:02d}/{epochs} --- Current avg. batch loss: {np.mean(losses):.4f}')

    def train_resnet(self, dataloader, epochs=20):
        self.vae = ResNetVAE(CNN_embed_dim=8)
        lr = np.linspace(0.005, 0.001, num=epochs)

        for epoch in range(epochs):
            opt = torch.optim.Adam(self.vae.parameters(), lr=lr[epoch])

            for idx, (x, y) in enumerate(dataloader):
                print(f'Batch nr. {idx + 1}/{len(dataloader)}')
                opt.zero_grad()
                x_hat, z, mu, sigma = self.vae(x)

                kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
                loss = ((x - x_hat) ** 2).sum() + kl

                loss.backward()
                opt.step()
            print(f'Epoch nr. {epoch + 1:02d}/{epochs} --- Current training loss: {loss:.4f}')

    def generate_new_dataset(self, output_size=100, img_dim=28):
        to_pil = T.ToPILImage()
        labels_array = np.zeros(output_size*len(self.labels))
        img_names = []

        for idx, label in enumerate(self.labels):
            print(f'\nGenerating data for label \'{label}\'')

            dataloader = self.get_data_by_label(label)

            # some classes have too few labels to train, skip those classes when generating data
            try:
                self.train_network(dataloader)
                torch.save(self.vae, f'models/variational_autoencoder_{label}.pt')
            except UnboundLocalError:
                continue

            U = torch.distributions.Uniform(-2, 2)
            for i in range(output_size):
                point = U.sample(sample_shape=[32, self.latent_dims])
                z = torch.Tensor(point)
                x_hat = self.vae.decoder(z).detach()[0]

                # x_hat = (((x_hat[:, :] - x_hat[:, :].min()) / (x_hat[:, :].max() - x_hat[:, :].min())) * 255)
                img = to_pil(x_hat)

                img.save(f'data/Generated/img/{label}_{i}.png')

                labels_array[output_size*idx+i] = label
                img_names.append(f'{label}_{i}.png')

            if self.latent_dims == 2:
                plots.plot_reconstructed(self.vae, f'reconstructed_{label}.png', r0=(-3, 3), r1=(-3, 3))

        try:
            labels_df = pd.DataFrame(labels_array, columns=['label'], index=None)
            labels_df['image_name'] = img_names
            labels_df.to_csv('data/GeneratedMNIST/labels.csv', index=False)
        except UnboundLocalError:
            pass



