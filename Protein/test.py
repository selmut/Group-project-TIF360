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
img_size = 128
output_size = 100

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train_reduced'

tfms = T.Compose([T.CenterCrop(img_size), T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float)])
tfms_back = T.Compose([T.ToPILImage()])

print('Loading data...')
dataset = ProteinData(data_dir+'/single_target_files.csv',
                      train_dir, transform=tfms)

gen = DataGenerator(dataset, 2)
loader = gen.get_data_by_label(0)

img = Image.open('data/Original/human-protein-atlas-image-classification/train_reduced/all_channels0.png')

img = tfms(img)
img = tfms_back(img)
img.show()
