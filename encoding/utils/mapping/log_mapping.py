import torch
import numpy as np

def linear2log(input_img, mu):
    '''
    perform hdr mapping: y = log(1 + \mu x) / log(1 + \mu)
    two fixed points: 0 -> 0 and 1 -> 1
    input: x in [0, +inf), mu in (0, +inf)
    output: y in [0, +inf) (compressed)
    '''
    if isinstance(input_img, np.ndarray):
        compressed_img = np.log(1 + input_img*mu) / np.log(1 + mu)
        return compressed_img
    elif isinstance(input_img, torch.Tensor):
        compressed_img = torch.log(1 + input_img*mu) / torch.log(torch.tensor(1 + mu, dtype=input_img.dtype))
        return compressed_img
    else:
        raise TypeError("assert input to be a torch.Tensor or numpy.ndarray!")


def log2linear(compressed_img, mu):
    '''
    map compressed log image back to linear hdr: x = (exp(y*(log(1 + \mu))) - 1) / \mu
    two fixed points: 0 -> 0 and 1 -> 1
    input: y in [0, +inf) (compressed), mu in (0, +inf)
    output: x in [0, +inf)
    '''
    if isinstance(compressed_img, np.ndarray):
        linear_img = (np.exp(compressed_img*(np.log(1+mu))) - 1) / mu
        return linear_img
    elif isinstance(compressed_img, torch.Tensor):
        linear_img = (torch.exp(compressed_img*(torch.log(torch.tensor(1 + mu, dtype=compressed_img.dtype)))) - 1) / torch.tensor(mu, dtype=compressed_img.dtype)
        return linear_img
    else:
        raise TypeError("assert input to be a torch.Tensor or numpy.ndarray!")
