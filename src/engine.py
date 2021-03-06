from preprocess import to_csv, cross_validation, get_data, Audio
from pathlib import Path
from dataset import SpeechCommandDataset
from dataloader import get_dl
import torch
import torch.nn as nn
import torch.optim as optim
from model import DNNModel, TCNModel, EdgeCRNN, DSCNN, LSTM, torch_to_tflite
from tqdm import tqdm
from train import train
from valid import valid
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
import hashlib


def run(cls, tfms, model_name='edgecrnn', epochs=5):
    md5 = hashlib.md5()
    with open('src/runtime_config.py', 'rb') as f:
        data = f.read()
        md5.update(data)
    req_hash = md5.hexdigest()

    file_list = os.listdir('data')
    req_files = [
        f'train_X_{req_hash}.pkl',
        f'train_y_{req_hash}.pkl',
        f'valid_X_{req_hash}.pkl',
        f'valid_y_{req_hash}.pkl',
        f'test_X_{req_hash}.pkl',
        f'test_y_{req_hash}.pkl'
    ]
    pkl_found = True
    for f in req_files:
        pkl_found &= (f in file_list)

    if not pkl_found:
        df = to_csv(cls)

        train_df, valid_df, test_df = cross_validation(df)
        train_X, train_y = get_data(train_df, tfms)
        valid_X, valid_y = get_data(valid_df, tfms)
        test_X, test_y = get_data(test_df, tfms)

        print(f'Saving training data...')
        with open(f'data/train_X_{req_hash}.pkl', 'wb') as f:
            pickle.dump(train_X, f)
        with open(f'data/train_y_{req_hash}.pkl', 'wb') as f:
            pickle.dump(train_y, f)
        print(f'Saving validation data...')
        with open(f'data/valid_X_{req_hash}.pkl', 'wb') as f:
            pickle.dump(valid_X, f)
        with open(f'data/valid_y_{req_hash}.pkl', 'wb') as f:
            pickle.dump(valid_y, f)
        print(f'Saving testing data...')
        with open(f'data/test_X_{req_hash}.pkl', 'wb') as f:
            pickle.dump(test_X, f)
        with open(f'data/test_y_{req_hash}.pkl', 'wb') as f:
            pickle.dump(test_y, f)
    else:
        print(f'Loading training data...')
        with open(f'data/train_X_{req_hash}.pkl', 'rb') as f:
            train_X = pickle.load(f)
        with open(f'data/train_y_{req_hash}.pkl', 'rb') as f:
            train_y = pickle.load(f)
        print(f'Loading validation data...')
        with open(f'data/valid_X_{req_hash}.pkl', 'rb') as f:
            valid_X = pickle.load(f)
        with open(f'data/valid_y_{req_hash}.pkl', 'rb') as f:
            valid_y = pickle.load(f)
        print(f'Loading testing data...')
        with open(f'data/test_X_{req_hash}.pkl', 'rb') as f:
            test_X = pickle.load(f)
        with open(f'data/test_y_{req_hash}.pkl', 'rb') as f:
            test_y = pickle.load(f)

    train_ds = SpeechCommandDataset(train_X, train_y)
    valid_ds = SpeechCommandDataset(valid_X, valid_y)
    test_ds = SpeechCommandDataset(test_X, test_y)

    train_dl = get_dl(train_ds, bs=16)
    valid_dl = get_dl(valid_ds, bs=32)
    test_dl = get_dl(test_ds, bs=32)

    loss_fn = nn.CrossEntropyLoss()
    if model_name == 'dnn':
        model = DNNModel(n_classes=len(cls))
    elif model_name == 'lstm':
        model = LSTM(n_labels=len(cls))
    elif model_name == 'dscnn':
        model = DSCNN(n_labels=len(cls))
    elif model_name == 'tcn':
        model = TCNModel(n_classes=len(cls), n_filters=32)
    elif model_name == 'edgecrnn':
        model = EdgeCRNN(n_class=len(cls))
    model = model.to('cuda')
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                            base_lr=1e-5,
                                            max_lr=1e-4,
                                            cycle_momentum=False)

    max_acc = 0
    writer = SummaryWriter()
    dummy_data = torch.randn((16, 39, 101)).to('cuda')
    writer.add_graph(model, dummy_data)

    for epoch in range(epochs):
        print(f'\nEpoch #{epoch}:')
        model = model.train()
        train(train_dl, model, loss_fn, optimizer,
              scheduler, writer, epoch, len(train_dl))
        valid_acc = valid(valid_dl, model, loss_fn, optimizer)
        writer.add_scalar('Metrics/validation_accuracy', valid_acc, epoch)
        print(f'Validation Accuracy: {valid_acc}')

        if valid_acc > max_acc:
            print(f'Saving model...')
            max_acc = valid_acc
            model = model.eval()
            torch.save(
                model.state_dict(),
                f'models/{model_name+str(int(max_acc*100))}.pt')
            torch_to_tflite(model, model_name + str(int(max_acc * 100)))

    writer.close()

    test_acc = valid(test_dl, model, loss_fn, optimizer)
    print(f'Test Accuracy: {test_acc}')
