import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from ClassMixedMNIST import MixedMNIST


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


def train(num_epochs, cnn, loaders):
    cnn.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.005, amsgrad=True)

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            output = cnn(b_x)[0]

            loss = criterion(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()  # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def test(cnn, loaders):
    # Test the model
    cnn.eval()
    accuracies = []
    with torch.no_grad():
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracies.append(accuracy)
            print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    return np.mean(accuracies)


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

        cnn = TestModel()

        num_epochs = 10

        train(num_epochs, cnn, {'train': train_loader})
        acc = test(cnn, {'test': test_loader})

        accuracies[n, idx] = acc
        print(f'Average accuracy: {acc:.4f}')

accuracies = np.mean(accuracies, axis=0)

plt.figure()
plt.plot(percentages, accuracies)
plt.savefig('img/accuracies.png')
plt.close()

df = pd.DataFrame(accuracies)
df.to_csv('csv/accuracies.csv', index=None, header=None)

