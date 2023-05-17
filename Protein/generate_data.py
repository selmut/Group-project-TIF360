import torchvision
import torchvision.transforms as T
import pandas as pd
import torch
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from ClassGenerator import DataGenerator
from ClassConvVAE import VariationalAutoencoder
from CustomDatasets.ClassProteinData import ProteinData

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train_reduced/'
labels_frame = pd.read_csv(data_dir+'/single_target_files.csv')

img_size = 256
latent_dims = 20
epochs = 10

tfms = T.Compose([T.CenterCrop(img_size), T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float)])
tfms_back = T.Compose([T.ToPILImage()])

print('Loading data...')
dataset = ProteinData(data_dir+'/single_target_files.csv',
                      train_dir, transform=tfms)

gen = DataGenerator(dataset, latent_dims)
loader = gen.generate_new_dataset(100)

'''vae = VariationalAutoencoder(latent_dims)
opt = torch.optim.Adam(vae.parameters())

for epoch in range(epochs):
    for idx, (x, y) in enumerate(loader):
        # print(f'Batch nr. {idx}/{len(loader)}')
        opt.zero_grad()
        x_hat = vae(x)
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
        loss.backward()
        opt.step()

    print(f'Epoch nr. {epoch + 1:02d}/{epochs} --- Current training loss: {loss:.4f}')

torch.save(vae, 'models/variational_autoencoder.pt')'''




