import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from ClassGeneratedMNIST import GeneratedMNIST
from ClassMixedMNIST import MixedMNIST

mixed = MixedMNIST()
dataloader = torch.utils.data.DataLoader(MixedMNIST(), batch_size=128, shuffle=True)

train_features, train_labels = next(iter(dataloader))

for idx in range(10):
    plt.figure()
    plt.imshow(train_features[idx])
    plt.show()

    print(train_labels[idx])



