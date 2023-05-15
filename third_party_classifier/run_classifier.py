import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from classifier import Classifier
from train_test_classifier import Train_Test
from MNIST.ClassMixedMNIST import MixedMNIST

n = 20
n_reals = 10
percentages = np.linspace(0, 1, num=n)
accuracies = np.zeros((n_reals, n))


for n in range(n_reals):
    for idx, p in enumerate(percentages):
        print(f'\nGenerated data percentage: {p}')
        original_data = MixedMNIST(dataset_size=60_000, percentage_generated=0)
        original_train, original_test = torch.utils.data.random_split(original_data, [50_000, 10_000])

        mix_data = MixedMNIST(dataset_size=60_000, percentage_generated=p)
        mix_train, mix_test = torch.utils.data.random_split(mix_data, [50_000, 10_000])

        train_loader = torch.utils.data.DataLoader(mix_train, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(original_test, batch_size=128, shuffle=True)

        cnn = Classifier()
        testModel = Train_Test(cnn)

        num_epochs = 10

        testModel.train(num_epochs, {'train': train_loader})
        acc = testModel.test({'test': test_loader})

        accuracies[n, idx] = acc
        print(f'Average accuracy: {acc:.4f}')

accuracies = np.mean(accuracies, axis=0)

plt.figure()
plt.plot(percentages, accuracies)
plt.savefig('img/accuracies.png')
plt.close()

df = pd.DataFrame(accuracies)
df.to_csv('csv/accuracies.csv', index=None, header=None)

