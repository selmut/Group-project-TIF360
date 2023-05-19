import numpy as np
import torch
import torch.nn as nn
from torchvision import io
import torchvision.transforms as T
from PIL import Image
from ClassConvVAE import VariationalAutoencoder
import pandas as pd
from CustomDatasets.ClassProteinData import ProteinData

img_size = 256
latent_dim = 10
epochs = 2
bs = 128

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train_rb/'
labels_frame = pd.read_csv(data_dir+'/single_target_files.csv')

tfms = T.Compose([T.CenterCrop(img_size), T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float)])
tfms_back = T.Compose([T.ToPILImage()])

dataset = ProteinData(data_dir+'/single_target_files.csv',
                      train_dir, transform=tfms)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=True, shuffle=True)

vae = VariationalAutoencoder(latent_dim)

opt = torch.optim.Adam(vae.parameters())
lrs = np.linspace(0.0005, 0.0001, num=len(dataloader)*epochs)

n = 0

last_epoch = 0

for epoch in range(epochs):
    for idx, (x, y) in enumerate(dataloader):
        opt = torch.optim.Adam(vae.parameters(), lr=lrs[n], amsgrad=True)
        opt.zero_grad()
        x_hat = vae(x)
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
        loss.backward()
        opt.step()

        print(f'Batch nr. {idx + 1}/{len(dataloader)}--- Current batch loss: {loss:.4f}')
        n += 1

    print(f'\nEpoch nr. {epoch + 1:02d}/{epochs} --- Current training loss: {loss:.4f}')
    last_epoch = epoch
    torch.save(vae, f'models/variational_autoencoder_general{epoch}.pt')

vae = torch.load(f'models/variational_autoencoder_general{last_epoch}.pt')
output_size = 10

to_pil = T.ToPILImage()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=True, shuffle=True)

U = torch.distributions.Uniform(-2, 2)
for i in range(output_size):
    point = U.sample(sample_shape=[bs, latent_dim])
    z = torch.Tensor(point)
    x_hat = vae.decoder(z).detach()[0]

    # x_hat = (((x_hat[:, :] - x_hat[:, :].min()) / (x_hat[:, :].max() - x_hat[:, :].min())) * 255)
    img = to_pil(x_hat)

    img.save(f'data/Generated/img/general_{i}.png')

