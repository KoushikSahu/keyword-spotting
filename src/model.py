import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import Audio
from pathlib import Path

class EdgeCRNN(nn.Module):
    def __init__(self):
        super(EdgeCRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=39,
                out_channels=39,
                kernel_size=3,
                stride=1,
                padding=1)

    def forward(self, inp):
        out = self.conv1(inp)
        print(out.shape)
        return out

if __name__ == '__main__':
    audio_path = Path('../data/google_speech_command/go/004ae714_nohash_0.wav')
    a = Audio(audio_path)
    model = EdgeCRNN()
    output = model(torch.tensor(torch.tensor(a.lfbe_deltadelta()).view(1, 39, -1)))
    print(output.shape)

