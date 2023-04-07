import torch
import utils
from config import Config
from nanoGPT_model import GPTLanguageModel
import time

max_iters = Config.max_iters
device = Config.device
learning_rate = Config.learning_rate
eval_interval = Config.eval_interval

data_path = 'data/romance.txt'

data_text, chars, vocab_size = utils.read_input_data(data_path)

stoi, itos = utils.get_stoi_itos(chars)

def encode(s): return [stoi[ch] for ch in s]
def decode(l): return ''.join([itos[i] for i in l])

data_torch = torch.tensor(encode(data_text), dtype=torch.long)

train, val = utils.split_data(data_torch, 0.7)

model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'parameters')

start = time.time()
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = utils.estimate_loss(model, train, val)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        time_checkpoint = time.time()
        print("\tElapsed time: ", time.strftime("%H:%M:%S", time.gmtime((time_checkpoint - start))))

    xb, yb = utils.get_batch(train)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save the model
torch.save(model, 'models/machado.pt')

# generate text from the model
context = torch.zeros((1,1), dtype=torch.long, device=device) # start with a single <BOS> token 

print(decode(model.generate(context, max_new_tokens=2500)[0].tolist()))     
     

# for i in range(2500):
#     # generate the next token using the entire previous context
#     next_token = model.generate(context, max_new_tokens=1)[0]
#     # decode and print the next token
#     print(decode(next_token.tolist()), end='')
#     # update the context with the next token
#     context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
