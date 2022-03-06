import librosa
from pathlib import Path
import numpy as np
import torch
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Compose():
  def __init__(self, tfms):
    self.tfms = tfms

  def __call__(self, inp):
    for fn in self.tfms:
        inp = fn(inp)
    return inp


class Audio():
  @staticmethod
  def load_audio(pth):
    data, sr = librosa.load(pth, sr=16000)
    if data.size < 16000:
      data = np.pad(data, (16000-data.size, 0), mode='constant')
    return {
        'data': np.array(data),
        'sr': sr
    }

  @staticmethod
  def mfcc(inp):
    feats = librosa.feature.mfcc(inp['data'], sr=16000, n_mfcc=39)
    return np.array(feats)

  @staticmethod
  def lfbe_delta(inp):
    mel_spec = librosa.feature.melspectrogram(inp['data'],
                                              sr=16000,
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
    return torch.tensor(inp)


def to_csv(clss, dset_pth=Path('data')):
  data_dict = dict()
  data_dict['filename'] = list()
  data_dict['classname'] = list()
  data_dict['class'] = list()

  for cls in clss:
    pth = dset_pth/cls
    for file in (itr := tqdm(os.listdir(pth))):
      data_dict['filename'].append(file)
      data_dict['classname'].append(cls)
      itr.set_description(f'{cls}/{file}')

  le = LabelEncoder()
  le.fit(data_dict['classname'])
  data_dict['class'] = le.transform(data_dict['classname'])

  return pd.DataFrame(data_dict)


def cross_validation(df):
  train_df, valid_df = train_test_split(
      df, test_size=0.2, stratify=df['class'], shuffle=True)
  valid_df, test_df = train_test_split(valid_df,
                                       test_size=0.5,
                                       stratify=valid_df['class'],
                                       shuffle=True)

  train_df = train_df.reset_index(drop=True)
  valid_df = valid_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  return train_df, valid_df, test_df


def get_data(df, tfms, base_pth=Path('data')):
  print('Getting data: ')
  X = list()
  y = list()

  for i in (itr := tqdm(df.iterrows())):
    filename = i[1]['filename']
    classname = i[1]['classname']
    cls = i[1]['class']

    pth = base_pth/classname/filename

    X.append(tfms(pth))
    y.append(torch.tensor(cls))
    itr.set_description(f'{classname}/{filename}')

  return X, y


if __name__ == '__main__':
  audio_path = Path('./data/speech_commands_v0.02/go/0132a06d_nohash_4.wav')
  tfms = Compose([Audio.load_audio, Audio.lfbe_delta, Audio.to_tensor])
  print(tfms(audio_path).shape)
