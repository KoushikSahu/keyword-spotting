from torch.utils.data import Dataset
from preprocess import Audio, Compose
import numpy as np
import torch
from pathlib import Path

class SpeechCommandDataset():
    def __init__(self, df, base_pth=Path('data/speech_commands_v0.02')):
        self.df = df
        self.base_pth = base_pth

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, 'filename']
        classname = self.df.loc[idx, 'classname']
        cls = self.df.loc[idx, 'class']

        tfms = Compose([Audio.load_audio, Audio.lfbe_delta, Audio.to_tensor])
        tfmd_ad = tfms(self.base_pth/classname/filename)

        return {
                'audio': tfmd_ad,
                'class': torch.tensor(cls)
                }

