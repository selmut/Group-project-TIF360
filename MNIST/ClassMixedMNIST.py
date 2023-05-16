import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from ClassGeneratedMNIST import GeneratedMNIST
import torchvision
from torch.utils.data import Dataset


class MixedMNIST(Dataset):
    def __init__(self, dataset_size=1_000, percentage_generated=0.5):
        self.original_dataset = torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(),
                                                           download=False)
        self.generated_dataset = GeneratedMNIST('data/GeneratedMNIST/labels.csv', 'data/GeneratedMNIST/img')
        self.percentage_generated = percentage_generated
        self.dataset_size = dataset_size
        self.data, self.targets = self.get_merged_dataset()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        image = image.reshape(-1, *image.shape)
        label = self.targets[idx]

        return image/255, label

    def get_splitting_idxs(self):
        idxs_original = np.array(list(range(self.original_dataset.__len__())))
        idxs_generated = np.array(list(range(self.generated_dataset.__len__())))

        np.random.shuffle(idxs_original)
        np.random.shuffle(idxs_generated)

        split_idx = int(self.dataset_size*self.percentage_generated)

        idxs_original = idxs_original[split_idx:]
        idxs_generated = idxs_generated[:split_idx]
        return idxs_original, idxs_generated

    def get_split_datasets(self):
        idxs_original, idxs_generated = self.get_splitting_idxs()

        original_data_split = self.original_dataset.data[idxs_original]
        generated_data_split = self.generated_dataset.data[idxs_generated]

        original_targets_split = self.original_dataset.targets[idxs_original]
        generated_targets_split = self.generated_dataset.targets[idxs_generated]

        return original_data_split, original_targets_split, generated_data_split, generated_targets_split

    def get_merged_dataset(self):
        original_data_split, original_targets_split, generated_data_split, generated_targets_split = self.get_split_datasets()

        data = torch.zeros((self.dataset_size, 28, 28))
        targets = torch.zeros(self.dataset_size)

        split_idx = int(self.dataset_size * self.percentage_generated)

        data[split_idx:] = original_data_split
        data[:split_idx] = generated_data_split

        targets[split_idx:] = original_targets_split
        targets[:split_idx] = generated_targets_split

        # TODO: maybe shuffle data+targets?
        return torch.tensor(data, dtype=torch.float), torch.tensor(targets, dtype=torch.long)



