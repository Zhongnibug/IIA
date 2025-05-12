from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

class EasyStepDecay:
    def __init__(self, optimizer, step_size=10, gamma=0.9):
        self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.optimizer = optimizer
    def __call__(self):
        self.scheduler.step()

class EasyCosine:
    def __init__(self, optimizer, max_epochs=20, eta_min=0, last_epoch=-1):
        self.scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min, last_epoch=last_epoch)
        self.optimizer = optimizer
    def __call__(self):
        self.scheduler.step()