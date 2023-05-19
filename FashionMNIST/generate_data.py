import torchvision
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from ClassGenerator import DataGenerator
from MNIST.ClassMixedMNIST import MixedMNIST

latent_dims = 8
nr_samples = 60_000

print('Loading data...')
dataset = torchvision.datasets.FashionMNIST('./data', transform=torchvision.transforms.ToTensor(), download=True)

# dataset = MixedMNIST(dataset_size=nr_samples, percentage_generated=0)

gen = DataGenerator(dataset, latent_dims)
gen.generate_new_dataset(output_size=6_000)





