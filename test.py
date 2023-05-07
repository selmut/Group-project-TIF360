import torch
import torchvision

print('Loading data...')
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False),
        batch_size=128, shuffle=True)

labels = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False).targets


