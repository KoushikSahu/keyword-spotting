from preprocess import to_csv, cross_validation, get_data
from pathlib import Path
from dataset import SpeechCommandDataset
from dataloader import get_dl
import torch
import torch.nn as nn
import torch.optim as optim
from model import DNNModel, TCNModel
from tqdm import tqdm
from train import train
from valid import valid

def run(cls, epochs=5):
  df = to_csv(cls)

  train_df, valid_df, test_df = cross_validation(df)
  train_X, train_y = get_data(train_df)
  valid_X, valid_y = get_data(valid_df)
  test_X, test_y = get_data(test_df)

  train_ds = SpeechCommandDataset(train_X, train_y)
  valid_ds = SpeechCommandDataset(valid_X, valid_y)
  test_ds = SpeechCommandDataset(test_X, test_y)

  train_dl = get_dl(train_ds, bs=16)
  valid_dl = get_dl(valid_ds, bs=32)
  test_dl = get_dl(test_ds, bs=32)

  loss_fn = nn.CrossEntropyLoss()
  model = TCNModel(n_classes=len(cls), n_filters=16).to('cuda')
  optimizer = optim.AdamW(model.parameters())

  for epoch in range(epochs):
    print(f'\nEpoch #{epoch}:')
    train(train_dl, model, loss_fn, optimizer)
    valid_acc = valid(valid_dl, model, loss_fn, optimizer)

    print(f'Validation Accuracy: {valid_acc}')

  test_acc = valid(test_dl, model, loss_fn, optimizer)
  print(f'Test Accuracy: {test_acc}')
