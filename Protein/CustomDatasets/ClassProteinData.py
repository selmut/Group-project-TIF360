from torchvision import io
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class ProteinData(Dataset):
    def __init__(self, csv_file, img_dir):
        self.ids = pd.read_csv(csv_file)['Id']
        self.labels = pd.read_csv(csv_file)['Target']
        self.img_dir = img_dir

        self.targets = self.get_targets()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        red = io.read_image(self.img_dir + self.ids[item] + '_red.png')
        green = io.read_image(self.img_dir + self.ids[item] + '_green.png')
        blue = io.read_image(self.img_dir + self.ids[item] + '_blue.png')
        yellow = io.read_image(self.img_dir + self.ids[item] + '_yellow.png')

        blue = torch.tensor(blue, dtype=torch.uint8).numpy()[0]
        red = torch.tensor(red, dtype=torch.uint8).numpy()[0]
        yellow = torch.tensor(yellow, dtype=torch.uint8).numpy()[0]
        green = torch.tensor(green, dtype=torch.uint8).numpy()[0]

        green = np.add(yellow, green)
        red = np.add(yellow, red)

        image = np.zeros((3, 512, 512))
        image[0, :, :] = red/255
        image[1, :, :] = green/255
        image[2, :, :] = blue/255

        return torch.tensor(image, dtype=torch.float), torch.tensor(self.labels.iloc[item], dtype=torch.long)

    def get_targets(self):
        targets = np.zeros(self.__len__())
        for n in range(self.__len__()):
            targets[n] = self.labels[n]

        return targets
