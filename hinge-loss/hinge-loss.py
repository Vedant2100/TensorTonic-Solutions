import numpy as np
import torch

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    
    y_true = torch.as_tensor(y_true, dtype = torch.float64) # not 32 bit as it gave errors
    y_score = torch.as_tensor(y_score, dtype = torch.float64)
    margin = torch.as_tensor(margin, dtype = torch.float64)
    pred = torch.clamp(margin - y_true * y_score, min = 0)
    if reduction == 'mean':
        return pred.mean().item()
    elif reduction == 'sum':
        return pred.sum().item()

    # Write code here
    pass