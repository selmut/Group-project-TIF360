import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable


class Train_Test:
    def __init__(self, cnn, lr=None):
        if lr is None:
            self.lr = 0.005
        else:
            self.lr = lr
        self.cnn = cnn

    def train(self, num_epochs, loaders):
        self.cnn.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr = self.lr, amsgrad=True)

        # Train the model
        total_step = len(loaders['train'])

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)  # batch x
                b_y = Variable(labels)  # batch y
                output = self.cnn(b_x)[0]

                loss = criterion(output, b_y)

                # clear gradients for this training step
                optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()  # apply gradients
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


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
