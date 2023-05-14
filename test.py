import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from ClassGeneratedMNIST import GeneratedMNIST
from ClassMixedMNIST import MixedMNIST

mixed = MixedMNIST()
dataloader = torch.utils.data.DataLoader(MixedMNIST(), batch_size=128, shuffle=True)




