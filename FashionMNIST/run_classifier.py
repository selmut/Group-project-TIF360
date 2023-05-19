import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ClassClassifier import Classifier
from ClassClassifier import TestModel
from ClassMixedMNIST import MixedMNIST
from ClassSampledMNIST import SampledMNIST

n = 10
n_reals = 1
percentages = np.linspace(0, 1, num=n)
accuracies = np.zeros(n)
nr_samples = 60_000
lr = 0.005
nr_epochs = 10

'''for idx, p in enumerate(percentages):
    print(f'\nGenerated data percentage: {p}')
    original_data = MixedMNIST(dataset_size=nr_samples, percentage_generated=0)
    original_train, original_test = torch.utils.data.random_split(original_data, [50_000, 10_000])

    mix_data = MixedMNIST(dataset_size=nr_samples, percentage_generated=p)
    mix_train, mix_test = torch.utils.data.random_split(mix_data, [50_000, 10_000])

    train_loader = torch.utils.data.DataLoader(mix_train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(original_test, batch_size=128, shuffle=True)

    cnn = Classifier()
    testModel = TestModel(cnn, train_loader, test_loader, lr, nr_epochs)
    testModel.train()
    acc = testModel.test()

    accuracies[idx] = acc
    print(f'Average accuracy: {acc:.4f}')

df = pd.DataFrame(accuracies)
df.to_csv('csv/accuracies_mixed.csv', index=None, header=None)'''

pool_sizes = np.linspace(100, 60_000, num=n)
accuracies = np.zeros(n)

for idx, p in enumerate(pool_sizes):
    print(f'\nSample pool size: {p}')
    original_data = MixedMNIST(dataset_size=nr_samples, percentage_generated=0)
    original_train, original_test = torch.utils.data.random_split(original_data, [50_000, 10_000])

    sampled_data = SampledMNIST(p)
    sampled_train, sampled_test = torch.utils.data.random_split(sampled_data, [50_000, 10_000])

    train_loader = torch.utils.data.DataLoader(sampled_train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(original_test, batch_size=128, shuffle=True)

    cnn = Classifier()
    testModel = TestModel(cnn, train_loader, test_loader, lr, nr_epochs)
    testModel.train()
    acc = testModel.test()

    accuracies[idx] = acc
    print(f'Average accuracy: {acc:.4f}')


df = pd.DataFrame(accuracies)
df.to_csv('csv/accuracies_sampled.csv', index=None, header=None)

