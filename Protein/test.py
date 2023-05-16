import numpy as np
import pandas as pd
from PIL import Image
from Protein.CustomDatasets.ClassProteinData import ProteinData

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train/'

labels_frame = pd.read_csv(data_dir+'/single_target_files.csv')
print(len(labels_frame))

'''data = ProteinData('data/Original/human-protein-atlas-image-classification/single_target_files.csv',
                   'data/Original/human-protein-atlas-image-classification/train/')

for n in range(100):
    img, label = data.__getitem__(n)
    if n == 0:
        print(label)

    pil_img = Image.fromarray(img.numpy().astype(np.uint8), mode='RGB')
    pil_img.save(f'img/all_channels{n}.png')'''

