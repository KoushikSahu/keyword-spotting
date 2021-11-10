from torch.utils.data import DataLoader

def get_dl(ds, bs, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

