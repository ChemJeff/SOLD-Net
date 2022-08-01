import torch
import numpy as np

def GammaTMO(hdr, gamma, e):
    if isinstance(hdr, np.ndarray):
        invgamma = 1.0 / gamma
        exposure = 2**e
        ldr = np.power(exposure * hdr, invgamma)
        ldr = np.clip(ldr, 0.0, 1.0)
    elif isinstance(hdr, torch.Tensor):
        invgamma = 1.0 / gamma
        exposure = 2**e
        ldr = torch.pow(exposure * hdr, invgamma)
        ldr = torch.clamp_max_(ldr, 1.0)
        ldr = torch.clamp_min_(ldr, 0.0)
    return ldr