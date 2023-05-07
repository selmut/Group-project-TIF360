import matplotlib.pyplot as plt
import torch
import torchvision

print('Loading data...')
'''data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False),
        batch_size=128, shuffle=True)'''

dataset = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(), download=False)

indices = [idx for idx, target in enumerate(dataset.targets) if target in [0]]
dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=128, drop_last=True)

for idx, batch in enumerate(dataloader):
    print(batch[0].shape)
    for elem in batch[0]:
        plt.figure()
        plt.imshow(elem.reshape((28, 28)))
        plt.show()
    break