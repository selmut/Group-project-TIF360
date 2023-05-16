import torchvision
import torch
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from ClassGenerator import DataGenerator
from ClassVAE import VariationalAutoencoder
from CustomDatasets.ClassProteinData import ProteinData

latent_dims = 2
img_dim = 512
epochs = 10

print('Loading data...')
dataset = ProteinData('data/Original/human-protein-atlas-image-classification/single_target_files.csv',
                      'data/Original/human-protein-atlas-image-classification/train/')

gen = DataGenerator(dataset, latent_dims, img_dim)
loader = gen.get_data_by_label(0)

vae = VariationalAutoencoder(latent_dims, img_dim)

opt = torch.optim.Adam(vae.parameters())
for epoch in range(epochs):
    for idx, (x, y) in enumerate(loader):
        print(f'Batch nr. {idx}/{len(loader)}')
        opt.zero_grad()
        x_hat = vae(x)
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
        loss.backward()
        opt.step()

    print(f'Epoch nr. {epoch + 1:02d}/{epochs} --- Current training loss: {loss:.4f}')

torch.save(vae, 'models/variational_autoencoder.pt')




