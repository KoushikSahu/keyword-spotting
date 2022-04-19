from re import S
from torch.utils.data import Dataset
from preprocess import Audio, Compose
import numpy as np
import torch
from pathlib import Path


class SpeechCommandDataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'audio': self.X[idx],
            'class': self.y[idx]
        }
