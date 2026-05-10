def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    import torch
    W = torch.as_tensor(W)
    L = math.sqrt(6/(fan_in+fan_out))
    W = W * 2
    W = W * L 
    W = W - L # inplace problematic with Long
    return W.numpy()
    # Write code here