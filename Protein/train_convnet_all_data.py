import torch
import torch.nn as nn
from torchvision import io
import torchvision.transforms as T
from PIL import Image
from ClassConvVAE import VariationalAutoencoder
import pandas as pd
from CustomDatasets.ClassProteinData import ProteinData

img_size = 256
latent_dim = 20
epochs = 10

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train_reduced/'
labels_frame = pd.read_csv(data_dir+'/single_target_files.csv')

tfms = T.Compose([T.CenterCrop(img_size), T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float)])
tfms_back = T.Compose([T.ToPILImage()])

dataset = ProteinData(data_dir+'/single_target_files.csv',
                      train_dir, transform=tfms)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, drop_last=True)

vae = VariationalAutoencoder(latent_dim)

opt = torch.optim.Adam(vae.parameters())

for epoch in range(epochs):
    for idx, (x, y) in enumerate(dataloader):
        print(f'Batch nr. {idx+1}/{len(dataloader)}')
        opt.zero_grad()
        x_hat = vae(x)
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
        loss.backward()
        opt.step()
    print(f'\nEpoch nr. {epoch + 1:02d}/{epochs} --- Current training loss: {loss:.4f}')
