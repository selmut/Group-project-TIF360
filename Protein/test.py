import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
from Protein.CustomDatasets.ClassProteinData import ProteinData
from ClassResnetVAE import ResNetVAE
from ClassGenerator import DataGenerator

epochs = 10
img_size = 256
output_size = 100

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train'

tfms = T.Compose([T.CenterCrop(img_size), T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float)])
tfms_back = T.Compose([T.ToPILImage()])

print('Loading data...')
dataset = ProteinData(data_dir+'/single_target_files.csv',
                      train_dir, transform=tfms)

gen = DataGenerator(dataset, 2, 1)
loader = gen.get_data_by_label(27)

for idx, (x, y) in enumerate(loader):
    img = tfms_back(x[0])
    img.save(f'img/{idx}.png')

# labels: 1, 4, 11?, 14, 21!!
