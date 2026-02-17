import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import pathlib

from mpl_toolkits.mplot3d import Axes3D 

try:
    import umap  
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

