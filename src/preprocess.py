import librosa
import librosa.display
from pathlib import Path
import numpy as np
import torch
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Compose():
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, inp):
        for fn in self.tfms:
            inp = fn(inp)
        return inp

class Audio():
    def __init__(self, path):
        self.data, self.sr = librosa.core.load(path)

    @staticmethod
    def lfbe_delta(inp):
        mel_spec = librosa.feature.melspectrogram(inp.data,
                16000,
                n_mels=13,
                hop_length=160,
                n_fft=480,
                fmin=20,
                fmax=4000)

        log_mel = librosa.core.power_to_db(mel_spec)

        lfbe_del = librosa.feature.delta(log_mel)
        lfbe_deldel = librosa.feature.delta(lfbe_del)
        features = np.vstack([log_mel, lfbe_del, lfbe_deldel])
        return np.array(features)

    @staticmethod
    def to_tensor(inp):
        if isinstance(inp, Audio):
            return torch.tensor(inp.data)
        return torch.tensor(inp)

def to_csv(clss, dset_pth):
    data_dict = dict()
    data_dict['filename'] = list()
    data_dict['class'] = list()

    for cls in clss:
        pth = dset_pth/cls
        for file in os.listdir(pth):
            data_dict['filename'].append(file)
            data_dict['class'].append(cls)

    return pd.DataFrame(data_dict)

def cross_validation(df):
    le = LabelEncoder()
    le.fit(df['class'])
    df['class'] = le.transform(df['class'])

    train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['class'], shuffle=True)
    valid_df, test_df = train_test_split(valid_df,
            test_size=0.5,
            stratify=valid_df['class'],
            shuffle=True)

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return le, train_df, valid_df, test_df

if __name__ == '__main__':
    audio_path = Path('../data/google_speech_command/go/004ae714_nohash_0.wav')
    a = Audio(audio_path)
    tfms = Compose([Audio.lfbe_delta, Audio.to_tensor])
    print(tfms(a))

    dset_pth = Path('../data/google_speech_command')
    to_csv(['yes', 'no'], dset_pth)

