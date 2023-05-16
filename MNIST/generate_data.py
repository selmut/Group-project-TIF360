import torchvision
import torch
import numpy as np
import plots
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from ClassGenerator import DataGenerator
from ClassMixedMNIST import MixedMNIST

latent_dims = 2
nr_samples = 10_000

print('Loading data...')
dataset = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False)


print(dataset.targets)
split1, split2 = torch.utils.data.random_split(dataset, [nr_samples, 60_000-nr_samples])
gen = DataGenerator(split1, latent_dims)
gen.generate_new_dataset(output_size=60_000)





