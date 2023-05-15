import pandas as pd

labels_frame = pd.read_csv('data/Original/human-protein-atlas-image-classification/train.csv')
invalid_targets = []

for idx, target in enumerate(labels_frame['Target']):
    if ' ' in list(target):
        invalid_targets.append(idx)

labels_frame = labels_frame.drop(invalid_targets, axis='index')
labels_frame.to_csv('data/Original/human-protein-atlas-image-classification/single_target_files.csv', index=None)


