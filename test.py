import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from ClassGeneratedMNIST import MergedMNIST

data = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False)

merged = MergedMNIST('data/GeneratedMNIST/labels.csv', 'data/GeneratedMNIST/img')

item = merged.__getitem__(0)
img = item['image'].numpy().reshape((28, 28))

dataset = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False)

targets = [target for idx, target in enumerate(dataset.targets)]
images = [img.flatten() for idx, img in enumerate(dataset.data)]

labels_df = pd.DataFrame(targets, columns=['label'])

print(dataset.targets.shape)

# dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=128, drop_last=True)

