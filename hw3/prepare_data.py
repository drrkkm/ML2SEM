
import torch
import numpy as np

np.random.seed(42)

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    DEVICE = torch.device('cuda')
else:
    from torch import FloatTensor, LongTensor
    DEVICE = torch.device('cpu')


BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'