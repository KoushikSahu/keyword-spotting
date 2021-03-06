from tqdm import tqdm
import torch


def train(dl, model, loss_fn, optimizer, scheduler, writer, epoch, total):
    curr_iter = 0
    for data in (itr := tqdm(dl)):
        # for dnn and tcn:
        # audio = data['audio'].to('cuda').view(-1, 1, 39, 101)
        # for edgecrnn:
        audio = data['audio'].to('cuda')
        targ = data['class'].to('cuda')

        optimizer.zero_grad()
        output = model(audio)
        loss = loss_fn(output, targ)
        writer.add_scalar("Loss/train", loss, epoch*total+curr_iter)
        curr_iter += 1
        loss.backward()
        optimizer.step()
        itr.set_description(f'Training Loss: {loss}')

        scheduler.step()
