import torch
import utils
from config import Config
from nanoGPT_model import GPTLanguageModel
import time
import sys
import os

assert os.path.exists('data'), "Please create a data folder in the root directory"
assert os.path.exists('models'), "Please create a models folder in the root directory"

assert len(sys.argv) == 3, "Usage: python train.py <data_name> <model_name> \n" \
                            "\t<data_name> is the name of the data in the data folder \n" \
                            "\t<model_name> is the output name of the model in the models folder\n" \
                            "\t e.g. python train.py machado.txt machado.pt"

assert sys.argv[1].endswith('.txt'), "Data file must be a .txt file"
assert sys.argv[2].endswith('.pt'), "Model file must be a .pt file"

data_name = sys.argv[1]
DATA_PATH = f"data/{data_name}"
MODEL_NAME = sys.argv[2]
model_path = f'models/{MODEL_NAME}'

max_iters = Config.max_iters
device = Config.device
learning_rate = Config.learning_rate
eval_interval = Config.eval_interval

data_text, chars, vocab_size = utils.read_input_data(DATA_PATH)

stoi, itos = utils.get_stoi_itos(chars)

def encode(s): return [stoi[ch] for ch in s]
def decode(l): return ''.join([itos[i] for i in l])

data_torch = torch.tensor(encode(data_text), dtype=torch.long)

train, val = utils.split_data(data_torch, 0.7)

model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'parameters')
print('-'*40)

start = time.time()
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = utils.estimate_loss(model, train, val)
        print(f"Epoch {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        time_checkpoint = time.time()
        print("\tElapsed time: ", time.strftime("%H:%M:%S", time.gmtime((time_checkpoint - start))))
        if iter == 0:
            print("\tEstimated time per epoch: ", time.strftime("%H:%M:%S", time.gmtime((time_checkpoint - start))))
        else:
            print("\tEstimated time remaining: ", time.strftime("%H:%M:%S", time.gmtime((time_checkpoint - start)*(max_iters-iter)/(iter+1))))
        print('-'*40)

    xb, yb = utils.get_batch(train)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete!")
print("\tTotal time: ", time.strftime("%H:%M:%S", time.gmtime((time.time() - start))), \
      "\n\tTime per epoch: ", time.strftime("%H:%M:%S", time.gmtime(((time.time() - start)/max_iters))))
print('-'*40 + "\n")

print("Saving model...\n")
# save the model
try:
    torch.save(model, model_path)
except:
    print("Could not save model.")

# generate text from the model
context = torch.zeros((1,1), dtype=torch.long, device=device) # start with a single <BOS> token 
model.eval()

max_new_tokens = 1000
loader = utils.Loader(f"Generating text with {max_new_tokens} characters...", "Done!", 0.1).start()
print("\n"+"-"*40)
print(decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))
print("-"*40)
loader.stop()
