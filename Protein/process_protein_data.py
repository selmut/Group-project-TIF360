import pandas as pd
import numpy as np
from torchvision import io
from PIL import Image
import torch

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train/'

labels_frame = pd.read_csv(data_dir+'/train.csv')
invalid_targets = []

for idx, target in enumerate(labels_frame['Target']):
    if ' ' in list(target):
        invalid_targets.append(idx)

labels_frame = labels_frame.drop(invalid_targets, axis='index')
ids = labels_frame['Id']

labels_frame['Filename'] = [f'all_channels{item}.png' for item in list(range(len(ids)))]
labels_frame.to_csv(data_dir+'/single_target_files.csv', index=None)

reduced_train_dir = data_dir + '/train_reduced/'

print(len(labels_frame))

targets = [int(target) for target in labels_frame['Target'].tolist()]
keys = list(set(targets))
counts = []

for key in keys:
    idx = np.where(np.array(targets) == key)[0]
    counts.append(len(idx))

counts_dict = dict(zip(keys, counts))
print(counts_dict)

'''for item in range(len(ids)):
    red = io.read_image(train_dir + ids.iloc[item] + '_red.png')
    green = io.read_image(train_dir + ids.iloc[item] + '_green.png')
    blue = io.read_image(train_dir + ids.iloc[item] + '_blue.png')
    yellow = io.read_image(train_dir + ids.iloc[item] + '_yellow.png')

    blue = torch.tensor(blue, dtype=torch.uint8).numpy()[0]
    red = torch.tensor(red, dtype=torch.uint8).numpy()[0]
    yellow = torch.tensor(yellow, dtype=torch.uint8).numpy()[0]
    green = torch.tensor(green, dtype=torch.uint8).numpy()[0]

    #green = np.add(yellow, green)
    #red = np.add(yellow, red)

    image = np.zeros((512, 512, 3))
    image[:, :, 0] = red
    image[:, :, 1] = green
    image[:, :, 2] = blue

    pil_img = Image.fromarray(image.astype(np.uint8), mode='RGB')
    pil_img.save(reduced_train_dir+f'all_channels{item}.png')'''
