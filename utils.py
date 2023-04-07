import torch
import nanoGPT_model

device = nanoGPT_model.Config.device
batch_size = nanoGPT_model.Config.batch_size
block_size = nanoGPT_model.Config.block_size
eval_iters = nanoGPT_model.Config.eval_iters

def get_batch(data):
    # split can be 'train' or 'val'
    # generate small random batch of data of input x and target y
    #data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    # create an x matrix of size (batch_size, block_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # create an y matrix of size (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    data = {'train': train_data, 'val': val_data}
    out = {}
    model.eval()
    for key in data:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[key])
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[key] = losses.mean()
    model.train()
    return out

def read_input_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_text = f.read()

    chars = sorted(list(set(data_text)))
    vocab_size = len(chars)

    return data_text, chars, vocab_size

def get_stoi_itos(chars):
    # dictionary mapping characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    # dictionary mapping integers to characters
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

def split_data(data_torch, split_ratio):
    # split the data into train and validation sets
    split = int(split_ratio * len(data_torch))
    train = data_torch[:split]
    val = data_torch[split:]
    return train, val