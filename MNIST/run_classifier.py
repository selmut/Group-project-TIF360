import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ClassClassifier import Classifier
from ClassClassifier import TestModel
from ClassMixedMNIST import MixedMNIST

n = 20
n_reals = 1
percentages = np.linspace(0, 1, num=n)
accuracies = np.zeros((n_reals, n))
nr_sampels = 60_000
lr = 0.005
nr_epochs = 10

for n in range(n_reals):
    for idx, p in enumerate(percentages):
        print(f'\nGenerated data percentage: {p}')
        original_data = MixedMNIST(dataset_size=nr_sampels, percentage_generated=0)
        original_train, original_test = torch.utils.data.random_split(original_data, [50_000, 10_000])

        mix_data = MixedMNIST(dataset_size=nr_sampels, percentage_generated=p)
        mix_train, mix_test = torch.utils.data.random_split(mix_data, [50_000, 10_000])

        train_loader = torch.utils.data.DataLoader(mix_train, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(original_test, batch_size=128, shuffle=True)

        cnn = Classifier()
        testModel = TestModel(cnn, train_loader, test_loader, lr, nr_epochs)
        testModel.train()
        acc = testModel.test()

        accuracies[n, idx] = acc
        print(f'Average accuracy: {acc:.4f}')

accuracies = np.mean(accuracies, axis=0)

plt.figure()
plt.plot(percentages, accuracies)
plt.savefig('img/accuracies.png')
plt.close()

df = pd.DataFrame(accuracies)
df.to_csv('csv/accuracies.csv', index=None, header=None)

