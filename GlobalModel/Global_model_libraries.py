import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
