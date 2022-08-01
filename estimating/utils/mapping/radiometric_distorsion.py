import torch
import numpy as np


def truncnormal(mu, sigma, shape, lower=None, upper=None):
    val = np.zeros(shape)
    val = val.reshape(-1)
    num_value = val.size
    num_ok = 0
    while num_ok < num_value:
        candidates = np.random.normal(mu, sigma, num_value)
        if lower is not None:
            candidates = candidates[candidates>=lower]
        if upper is not None:
            candidates = candidates[candidates<=upper]
        num_used = min(num_value - num_ok, candidates.size)
        val[num_ok:num_ok+num_used] = candidates[:num_used]
        num_ok += num_used
    val = val.reshape(shape)
    return val

def trunclognormal(mu, sigma, shape, lower=None, upper=None):
    val = np.zeros(shape)
    val = val.reshape(-1)
    num_value = val.size
    num_ok = 0
    while num_ok < num_value:
        candidates = np.random.lognormal(mu, sigma, num_value)
        if lower is not None:
            candidates = candidates[candidates>=lower]
        if upper is not None:
            candidates = candidates[candidates<=upper]
        num_used = min(num_value - num_ok, candidates.size)
        val[num_ok:num_ok+num_used] = candidates[:num_used]
        num_ok += num_used
    val = val.reshape(shape)
    return val

def RadiometricDistorsion(input_img,
     exp_mean=0.2, exp_std=np.sqrt(0.2), exp_lower=0.1, exp_upper=10.0,
     whl_mean=1.0, whl_std=0.06, whl_lower=0.8, whl_upper=1.2,
      gma_mean=0.0035, gma_std=np.sqrt(0.2), gma_lower=0.85, gma_upper=1.2):
    # assume input image is BCHW
    B, C, H, W = input_img.shape
    exp_distortion = torch.Tensor(trunclognormal(exp_mean, exp_std, (B, 1, 1, 1), exp_lower, exp_upper)).to(input_img.device)
    whl_distortion = torch.Tensor(truncnormal(whl_mean, whl_std, (B, C, 1, 1), whl_lower, whl_upper)).to(input_img.device)
    gma_distortion = torch.Tensor(trunclognormal(gma_mean, gma_std, (B, 1, 1, 1), gma_lower, gma_upper)).to(input_img.device)
    distorted_img = input_img
    distorted_img = distorted_img * exp_distortion
    distorted_img = distorted_img * whl_distortion
    distorted_img = torch.pow(distorted_img, 1/gma_distortion)
    return distorted_img, (exp_distortion, whl_distortion, gma_distortion)

def GetDistortConfig(input_img,
     exp_mean=0.2, exp_std=np.sqrt(0.2), exp_lower=0.1, exp_upper=10.0,
     whl_mean=1.0, whl_std=0.06, whl_lower=0.8, whl_upper=1.2,
      gma_mean=0.0035, gma_std=np.sqrt(0.2), gma_lower=0.85, gma_upper=1.2):
    # assume input image is BCHW
    B, C, H, W = input_img.shape
    exp_distortion = torch.Tensor(trunclognormal(exp_mean, exp_std, (B, 1, 1, 1), exp_lower, exp_upper)).to(input_img.device)
    whl_distortion = torch.Tensor(truncnormal(whl_mean, whl_std, (B, C, 1, 1), whl_lower, whl_upper)).to(input_img.device)
    gma_distortion = torch.Tensor(trunclognormal(gma_mean, gma_std, (B, 1, 1, 1), gma_lower, gma_upper)).to(input_img.device)
    return exp_distortion, whl_distortion, gma_distortion

def DistortImage(input_img, exp_distortion=None, whl_distortion=None, gma_distortion=None):
    distorted_img = input_img
    if exp_distortion is not None:
        distorted_img = distorted_img * exp_distortion
    if whl_distortion is not None:
        distorted_img = distorted_img * whl_distortion
    if gma_distortion is not None:
        distorted_img = torch.pow(distorted_img, 1/gma_distortion)
    return distorted_img