import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, device, criterion, optimizer=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        
    def __call__(self, preds, labels, valid_len):

        x_ = preds.to(self.device).contiguous().view(-1, preds.size(-1))
        y_ = labels.to(self.device).contiguous().view(-1)
        loss = self.criterion(x_, y_)
        loss /= valid_len
        if self.optimizer is not None:
            self.optimizer(loss)
        return loss.detach().item() * valid_len
    
class IncrementAwareLossCompute:
    def __init__(self, device, criterion, optimizer=None,
                 low_weight=1.0,
                 high_weight=1.0):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.low_weight = low_weight
        self.high_weight = high_weight
        
    def __call__(self, low_preds, low_labels, low_valid_len,
                 high_preds, high_labels, high_valid_len):

        low_x_ = low_preds.to(self.device).contiguous().view(-1, low_preds.size(-1))
        low_y_ = low_labels.to(self.device).contiguous().view(-1)
        high_x_ = high_preds.to(self.device).contiguous().view(-1, high_preds.size(-1))
        high_y_ = high_labels.to(self.device).contiguous().view(-1)
        low_loss = self.criterion(low_x_, low_y_)
        high_loss = self.criterion(high_x_, high_y_)
        low_loss /= low_valid_len
        high_loss /= high_valid_len
        loss = low_loss * self.low_weight + high_loss * self.high_weight
        if self.optimizer is not None:
            self.optimizer(loss)
        return low_loss.detach().item() * self.low_weight * low_valid_len + high_loss.detach().item() * self.high_weight * high_valid_len, \
            low_loss.detach().item() * self.low_weight * low_valid_len, \
            high_loss.detach().item() * self.high_weight * high_valid_len