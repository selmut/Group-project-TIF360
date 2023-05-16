from torch.utils.data import Dataset, Subset, RandomSampler, SubsetRandomSampler
import numpy as np
import torch
import torchvision


class SampledMNIST(Dataset):
    def __init__(self, sample_pool_size, dataset_size=60_000):
        self.original_dataset = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(),
                                                          download=False)
        self.dataset_size = dataset_size
        self.sample_pool_size = sample_pool_size

        self.data, self.labels = self.get_samples_from_pool()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = self.data[item]
        image = image.reshape(-1, *image.shape)
        label = self.labels[item]

        return image/255, label

    def get_samples_from_pool(self):
        indexes = list(range(self.sample_pool_size))
        subset_of_original = Subset(self.original_dataset, indexes)
        sampled_indexes = RandomSampler(subset_of_original, num_samples=self.dataset_size, replacement=True)

        sampled_data = torch.zeros((self.dataset_size, 28, 28))
        sampled_labels = torch.zeros(self.dataset_size)

        for i, sample_idx in enumerate(sampled_indexes):
            sampled_data[i] = self.original_dataset.data[sample_idx]
            sampled_labels[i] = self.original_dataset.targets[sample_idx]

        return sampled_data, sampled_labels

