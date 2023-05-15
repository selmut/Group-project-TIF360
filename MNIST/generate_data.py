import torchvision
import numpy as np
import plots
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200


from ClassGenerator import DataGenerator

latent_dims = 2

print('Loading data...')
dataset = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False)

gen = DataGenerator(dataset, latent_dims)
gen.generate_new_dataset(output_size=60_00)





