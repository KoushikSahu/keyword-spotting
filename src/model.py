import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import Audio
from pathlib import Path
from tsai.all import TCN

class TCNModel(nn.Module):
  def __init__(self, n_classes, n_filters):
    super(TCNModel, self).__init__()
    self.tcn = TCN(c_in=39, c_out=n_classes, layers=[25]*n_filters)

  def forward(self, inp):
    return self.tcn(inp)

class DNNModel(nn.Module):
  def __init__(self, n_classes):
    super(DNNModel, self).__init__()
    self.n_classes = n_classes
    self.lin1 = nn.Linear(in_features=39*101, out_features=512)
    self.lin2 = nn.Linear(in_features=512, out_features=n_classes)

  def forward(self, inp):
    inp = torch.flatten(inp, start_dim=1)
    inp = F.relu(self.lin1(inp))
    out = self.lin2(inp)

    return out
