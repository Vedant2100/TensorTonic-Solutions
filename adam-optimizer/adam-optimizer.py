import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = np.array(m)
    grad = np.array(grad)
    v = np.array(v)
    param = np.array(param)
    m_new = beta1*m + (1-beta1)*grad
    v_new = beta2*v + (1-beta2)*(grad**2)
    param_new = param - lr * (m_new / (1 - beta1**t))/(np.sqrt(v_new/(1-beta2**t)) + eps)
    return (param_new, m_new, v_new)
    

