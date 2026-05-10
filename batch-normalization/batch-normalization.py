import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    import torch
    import math
    # print(x.shape) list
    x = torch.as_tensor(x).float()
    gamma = torch.as_tensor(gamma).float()
    beta = torch.as_tensor(beta).float()
    

    # --------------------------------
    if x.ndim == 2:
        mean = torch.mean(x, dim = 0) # over the batch
        var = torch.var(x, dim = 0, unbiased = False)
        x_new = (x - mean)/(torch.sqrt(var + eps))
    elif x.ndim == 4:
        C = x.shape[1]
        mean = torch.mean(x, dim = [0,2,3], keepdim =True) # over the batch
        var = torch.var(x, dim = [0,2,3], keepdim =True, unbiased = False)
        gamma = gamma.reshape(1, C, 1, 1)
        beta = beta.reshape(1, C, 1, 1)
        x_new = (x - mean)/(torch.sqrt(var + eps))
    return gamma* x_new+ beta #not @ , we need HADAMARD
    pass