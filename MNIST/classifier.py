import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = data.view(data.size(0), -1)
        output = self.out(data)
        return output, data


class TestModel:
    def __init__(self, cnn, train_loader, test_loader, lr=0.005, num_epochs=10):
        self.lr = lr
        self.cnn = cnn
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.lr, amsgrad=True)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

    def train(self):
        self.cnn.train()

        # Train the model
        total_step = len(self.train_loader)

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)  # batch x
                b_y = Variable(labels)  # batch y
                output = self.cnn(b_x)[0]

                loss = self.criterion(output, b_y)

                # clear gradients for this training step
                self.optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()  # apply gradients
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item()))

    def test(self, loaders):
        # Test the model
        self.cnn.eval()
        accuracies = []
        with torch.no_grad():
            for images, labels in loaders['test']:
                test_output, last_layer = self.cnn(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                accuracies.append(accuracy)
                print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
        return np.mean(accuracies)
