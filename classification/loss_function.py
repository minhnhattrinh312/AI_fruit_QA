import torch
import torch.nn as nn
import torch.nn.functional as F

class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target)