import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F

class LabelSmoothingBatchSum(nn.Module):
    "Implement label smoothing."
    def __init__(self, pad_idx, size, smoothing=0.0, reduction="sum"):
        super(LabelSmoothingBatchSum, self).__init__()
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - self.smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(-1) == self.size
        true_dist = x.data.clone().detach()
        true_dist_original_shape = x.shape
        true_dist = true_dist.view(-1, true_dist.size(-1))
        target = target.view(-1)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(-1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        true_dist = true_dist.view(true_dist_original_shape)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)