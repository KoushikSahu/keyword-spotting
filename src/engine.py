from preprocess import to_csv, cross_validation
from pathlib import Path
from dataset import SpeechCommandDataset
from dataloader import get_dl

def dnn():
    base_pth = Path('data/google_speech_command')
    df = to_csv(['yes', 'no'], base_pth)

    le, train_df, valid_df, test_df = cross_validation(df)

    train_ds = SpeechCommandDataset(train_df, le, base_pth)
    valid_ds = SpeechCommandDataset(valid_df, le, base_pth)
    test_ds = SpeechCommandDataset(test_df, le, base_pth)

    train_dl = get_dl(train_ds, bs=16)
    valid_dl = get_dl(valid_ds, bs=32)
    test_dl = get_dl(test_ds, bs=32)

    for data in train_dl:
        print(data)
        break

