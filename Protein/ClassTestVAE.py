import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn as nn
from torchvision import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torchvision.models as models
import torch.nn.functional as F
from tqdm.notebook import tqdm

from sklearn.metrics import f1_score

data_dir = 'data/Original/human-protein-atlas-image-classification'
train_dir = data_dir + '/train_reduced'
df = pd.read_csv(data_dir+'/single_target_files.csv')


labels = {0: 'Nucleoplasm', 1: 'Nuclear membrane', 2: 'Nucleoli', 3: 'Nucleoli fibrillar center', 4: 'Nuclear speckles',
          5: 'Nuclear bodies', 6: 'Endoplasmic reticulum', 7: 'Golgi apparatus', 8: 'Peroxisomes', 9: 'Endosomes',
          10: 'Lysosomes', 11: 'Intermediate filaments', 12: 'Actin filaments', 13: 'Focal adhesion sites',
          14: 'Microtubules', 15: 'Microtubule ends', 16: 'Cytokinetic bridge', 17: 'Mitotic spindle',
          18: 'Microtubule organizing center', 19: 'Centrosome', 20: 'Lipid droplets', 21: 'Plasma membrane',
          22: 'Cell junctions', 23: 'Mitochondria', 24: 'Aggresome', 25: 'Cytosol', 26: 'Cytoplasmic bodies',
          27: 'Rods & rings'}


class HumanProteinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, label = row['Filename'], row['Target']
        img_path = self.img_dir + "/" + str(img_id)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label


batch_size = 64
image_size = 256

tfms = T.Compose([
    T.Resize(image_size),
    T.ToTensor()
    # T.Normalize(*stats,inplace=True)
])

dataset = HumanProteinDataset(df, train_dir, transform=tfms)
train_ds, val_ds = torch.utils.data.random_split(dataset, [int(0.9*len(df)), len(dataset)-int(0.9*len(df))])

# img, label = dataset.__getitem__(np.random.randint(0, len(df)))

train_dl = DataLoader(train_ds, 2*batch_size)
val_dl = DataLoader(val_ds, 2*batch_size)


def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(17, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        data = 1 - images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        plt.savefig('img/batch_grid.png')
        break


# show_batch(train_dl, invert=False)


def f_score(labels, targets, threshold=0.5):
    labels = labels > threshold
    targets = targets > threshold

    TP = (labels & targets).sum(1).float()
    FP = (labels & (~targets)).sum(1).float()
    TN = ((~labels) & (~targets)).sum(1).float()
    FN = ((~labels) & targets).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-10))
    recall = torch.mean(TP / (TP + FN + 1e-10))
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1_score.mean(0)


class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)  # Generate predictions
        loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        score = f_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_score']))


resnet34 = models.resnet34(pretrained=True)

resnet34.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(resnet34.fc.in_features, 128),
    nn.Linear(128, 10)
)


class ProteinResnet(MultilabelImageClassificationBase):
    def __init__(self, resnet34):
        super().__init__()
        self.resnet34 = resnet34

    def forward(self, xb):
        return torch.sigmoid(self.resnet34(xb))


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device=device)
val_dl = DeviceDataLoader(val_dl, device=device)

model = to_device(ProteinResnet(resnet34), device)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


history = [evaluate(model, val_dl)]
print(history)
