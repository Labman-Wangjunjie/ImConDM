import torch
from torch.utils.data import DataLoader

import numpy as np

datapath = './train/machine-1-1'
datasets = np.loadtxt(datapath,delimiter=',')