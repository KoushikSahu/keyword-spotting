from tqdm import tqdm
import torch

def train(dl, model, loss_fn, optimizer):
  for data in (itr := tqdm(dl)):
    audio = data['audio'].to('cuda').view(-1, 39*101)
    targ = data['class'].to('cuda')

    optimizer.zero_grad()
    output = model(audio)
    loss = loss_fn(output, targ)
    loss.backward()
    optimizer.step()
    itr.set_description(f'Training Loss: {loss}')
