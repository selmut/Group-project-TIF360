import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import io
import torch


class GeneratedMNIST(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.data = self.get_data()
        self.targets = self.get_targets()

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.labels_frame['image_name'].iloc[idx]))
        image = io.read_image(img_name)
        image = torch.tensor(image)/255

        label = self.labels_frame['label'].iloc[idx]
        sample = {'image': image, 'label': label}
        return sample

    def get_data(self):
        out = np.zeros((self.__len__(), 28, 28))

        for idx in range(self.__len__()):
            item = self.__getitem__(idx)
            out[idx, :, :] = item['image'].numpy().reshape((28, 28))
        return torch.tensor(out)

    def get_targets(self):
        out = np.zeros(self.__len__())

        for idx in range(self.__len__()):
            item = self.__getitem__(idx)
            out[idx] = item['label']
        return torch.tensor(out)
