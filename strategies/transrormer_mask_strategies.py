import torch
import numpy as np

def subsequent_mask_3d(batch_size, size):
    "Mask out subsequent positions."
    attn_shape = (batch_size, size, size)
    
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 1

def subsequent_mask_2d(size):
    "Mask out subsequent positions."
    attn_shape = (size, size)
    
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 1