import torch
import utils
from config import Config
import nanoGPT_model
import __main__
import sys
import os

assert os.path.exists('data'), "Please ccheck if there is a data folder in the root directory with data"
assert os.path.exists('models'), "Please check if there is a models folder in the root directory with models"

assert len(sys.argv) == 3, "Usage: python inference.py <data_name> <model_name> <numbers_of_token>" \
                            "\t<data_name> is the name of the data in the data folder \n" \
                            "\t<model_name> is the name of the model in the models folder\n" \
                            "\t<numbers_of_token> is the number of token to generate\n" \
                            "\t e.g. python inference.py machado.txt machado.pt 2500"

assert sys.argv[1].endswith('.txt'), "Data file must be a .txt file"
assert sys.argv[2].endswith('.pt'), "Model file must be a .pt file"
assert isinstance(int(sys.argv[3]), int), "Number of tokens must be an integer"

DATA_NAME = sys.argv[1]
DATA_PATH = f'data/{DATA_NAME}'
MODEL_NAME = sys.argv[2]
MODEL_PATH = f'models/{MODEL_NAME}'
NUM_TOKENS = int(sys.argv[3])

data_text, chars, vocab_size = utils.read_input_data(DATA_PATH)
stoi, itos = utils.get_stoi_itos(chars)
def decode(l): return ''.join([itos[i] for i in l])

# Hack to load the model trained in another file
setattr(__main__, 'GPTLanguageModel', nanoGPT_model.GPTLanguageModel)
setattr(__main__, 'Block', nanoGPT_model.Block)
setattr(__main__, 'MultiHeadAttention', nanoGPT_model.MultiHeadAttention)
setattr(__main__, 'Head', nanoGPT_model.Head)
setattr(__main__, 'FeedForward', nanoGPT_model.FeedForward)

model = torch.load(MODEL_PATH).to(Config.device)
model.eval()

context = torch.zeros((1,1), dtype=torch.long, device=Config.device) # start with a single <BOS> token 

print(decode(model.generate(context, max_new_tokens=NUM_TOKENS)[0].tolist())) 


# for i in range(2500):
#     # generate the next token using the entire previous context
#     next_token = model.generate(context, max_new_tokens=1)[0]
#     # decode and print the next token
#     print(decode(next_token.tolist()), end='')
#     # update the context with the next token
#     context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
