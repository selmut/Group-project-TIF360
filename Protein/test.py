import numpy as np
from torchvision import io
import torch
from PIL import Image
from ClassProteinData import ProteinData

data = ProteinData('data/Original/human-protein-atlas-image-classification/single_target_files.csv',
                   'data/Original/human-protein-atlas-image-classification/train/')

for n in range(100):
    img, label = data.__getitem__(n)

    pil_img = Image.fromarray(img.numpy().astype(np.uint8), mode='RGB')
    pil_img.save(f'img/all_channels{n}.png')

