from torchvision import io
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class ProteinData(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.targets = self.get_targets()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id, label = row['Filename'], row['Target']
        img_path = self.img_dir + "/" + str(img_id)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def get_targets(self):
        targets = np.zeros(self.__len__())
        for n in range(self.__len__()):
            targets[n] = self.df['Target'][n]

        return targets
