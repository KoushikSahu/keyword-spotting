from torch.utils.data import Dataset
from preprocess import Audio, Compose
import numpy as np
import torch

class SpeechCommandDataset():
    def __init__(self, df, le, base_pth):
        self.df = df
        self.le = le
        self.base_pth = base_pth

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'filename']
        cls = int(self.df.loc[idx, 'class'])
        cls_name = self.le.inverse_transform([cls])[0]

        ad = Audio(self.base_pth/cls_name/filename)
        tfms = Compose([Audio.lfbe_delta, Audio.to_tensor])
        tfmd_ad = tfms(ad)

        return {
                'audio': tfmd_ad,
                'class': torch.tensor(cls)
                }

