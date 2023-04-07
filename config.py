import torch


class Config:
    # number of sequences in a batch (we can process batches in parallel)
    batch_size = 64
    # what is the maximum context length for predictions? / length of a sequence (maximum context lengt)
    block_size = 256
    # how many iterations to train the model 
    max_iters = 2500
    # evaluate the model every eval_interval iterations
    eval_interval = 500
    # learning rate
    learning_rate = 3e-4
    # 'cuda' (GPU) or 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # number of iterations to evaluate the model
    eval_iters = 250
    # number of channels in the hidden state
    n_embd = 384
    # number of heads in the multi-head attention mechanism
    n_head = 6
    # number of transformer blocks
    n_layer = 6
    # dropout rate
    dropout = 0.2
    
    # Check device (CPU or 'cuda' (GPU)) (GPU is recommended)
    print('device:', device)
    
    # Check if CUDA is correct installed
    # print(torch.zeros(1).cuda())
