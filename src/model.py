import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import Audio
from pathlib import Path

class DNN(nn.Module):
  def __init__(self):
    super(DNN, self).__init__()
    self.lin1 = nn.Linear(in_features=39*101, out_features=512)
    self.lin2 = nn.Linear(in_features=512, out_features=2)

  def forward(self, inp):
    inp = F.relu(self.lin1(inp))
    out = self.lin2(inp)

    return out
