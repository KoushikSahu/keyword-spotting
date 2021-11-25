from preprocess import to_csv, cross_validation, get_data
from pathlib import Path
from dataset import SpeechCommandDataset
from dataloader import get_dl
import torch
import torch.nn as nn
import torch.optim as optim
from model import DNNModel, TCNModel, EdgeCRNN
from tqdm import tqdm
from train import train
from valid import valid
import os
import pickle

def run(cls, epochs=5):
  file_list = os.listdir('data')
  req_files = [
    'train_X.pkl',
    'train_y.pkl',
    'valid_X.pkl',
    'valid_y.pkl',
    'test_X.pkl',
    'test_y.pkl'
  ]
  pkl_found = True
  for f in req_files:
    pkl_found &= (f in file_list)

  if not pkl_found:
    df = to_csv(cls)

    train_df, valid_df, test_df = cross_validation(df)
    train_X, train_y = get_data(train_df)
    valid_X, valid_y = get_data(valid_df)
    test_X, test_y = get_data(test_df)

    with open('data/train_X.pkl', 'wb') as f:
      pickle.dump(train_X, f)
    with open('data/train_y.pkl', 'wb') as f:
      pickle.dump(train_y, f)
    with open('data/valid_X.pkl', 'wb') as f:
      pickle.dump(valid_X, f)
    with open('data/valid_y.pkl', 'wb') as f:
      pickle.dump(valid_y, f)
    with open('data/test_X.pkl', 'wb') as f:
      pickle.dump(test_X, f)
    with open('data/test_y.pkl', 'wb') as f:
      pickle.dump(test_y, f)
  else:
    with open('data/train_X.pkl', 'rb') as f:
      train_X = pickle.load(f)
    with open('data/train_y.pkl', 'rb') as f:
      train_y = pickle.load(f)
    with open('data/valid_X.pkl', 'rb') as f:
      valid_X = pickle.load(f)
    with open('data/valid_y.pkl', 'rb') as f:
      valid_y = pickle.load(f)
    with open('data/test_X.pkl', 'rb') as f:
      test_X = pickle.load(f)
    with open('data/test_y.pkl', 'rb') as f:
      test_y = pickle.load(f)

  train_ds = SpeechCommandDataset(train_X, train_y)
  valid_ds = SpeechCommandDataset(valid_X, valid_y)
  test_ds = SpeechCommandDataset(test_X, test_y)

  train_dl = get_dl(train_ds, bs=16)
  valid_dl = get_dl(valid_ds, bs=32)
  test_dl = get_dl(test_ds, bs=32)

  loss_fn = nn.CrossEntropyLoss()
  model = EdgeCRNN(width_mult=1.).to('cuda')
  optimizer = optim.AdamW(model.parameters())
  scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                            base_lr=1e-5,
                                            max_lr=1e-4,
                                            cycle_momentum=False)

  for epoch in range(epochs):
    print(f'\nEpoch #{epoch}:')
    train(train_dl, model, loss_fn, optimizer, scheduler)
    valid_acc = valid(valid_dl, model, loss_fn, optimizer)

    print(f'Validation Accuracy: {valid_acc}')

  test_acc = valid(test_dl, model, loss_fn, optimizer)
  print(f'Test Accuracy: {test_acc}')
