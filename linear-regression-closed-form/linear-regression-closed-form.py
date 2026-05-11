import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    import torch
    X = torch.as_tensor(X, dtype= torch.float32)
    X_tr = torch.as_tensor(X.T, dtype=torch.float32)
    y = torch.as_tensor(y, dtype = torch.float32)
    w = torch.inverse(X_tr@ X)@X_tr@y
    return w.numpy()
    pass