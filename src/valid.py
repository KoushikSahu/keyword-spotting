from tqdm import tqdm
import torch

def valid(dl, model, loss_fn, optimizer):
  correct = 0
  total = 0
  
  with torch.no_grad():
    for data in (itr := tqdm(dl)):
      audio = data['audio'].to('cuda').view(-1, 39*101)
      targ = data['class'].to('cuda')

      output = model(audio)
      loss = loss_fn(output, targ)
      itr.set_description(f'Validation Loss: {loss}')

      preds = torch.argmax(output, dim=1)
      correct += (preds==targ).sum()
      total += targ.shape[0]

  accuracy = correct / total

  return accuracy
