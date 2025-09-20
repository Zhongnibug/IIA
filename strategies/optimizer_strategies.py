import torch
import torch.nn as nn
import torch.optim as optim
from utils.get_attr import no_attr

class ADAMoptimizer:
    def __init__(self, model, learning_rate, weight_decay=0.0):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
        self.model = model
    def __call__(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        pass
