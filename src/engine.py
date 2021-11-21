from preprocess import to_csv, cross_validation
from pathlib import Path
from dataset import SpeechCommandDataset
from dataloader import get_dl
import torch
import torch.nn as nn
import torch.optim as optim
from model import DNN
from tqdm import tqdm
from train import train
from valid import valid

def dnn(epochs=5):
  df = to_csv(['yes', 'no'])

  train_df, valid_df, test_df = cross_validation(df)

  train_ds = SpeechCommandDataset(train_df)
  valid_ds = SpeechCommandDataset(valid_df)
  test_ds = SpeechCommandDataset(test_df)

  train_dl = get_dl(train_ds, bs=16)
  valid_dl = get_dl(valid_ds, bs=32)
  test_dl = get_dl(test_ds, bs=32)

  loss_fn = nn.CrossEntropyLoss()
  model = DNN().to('cuda')
  optimizer = optim.AdamW(model.parameters())

  for epoch in range(epochs):
    train(train_dl, model, loss_fn, optimizer)
    valid_acc = valid(valid_dl, model, loss_fn, optimizer)

    print(f'Validation Accuracy: {valid_acc}')

  test_acc = valid(test_dl, model, loss_fn, optimizer)
  print(f'Test Accuracy: {test_acc}')
