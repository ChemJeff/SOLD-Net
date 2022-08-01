import numpy as np
import cv2

import torch
import torch.nn as nn

EPS = 1e-12

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

def mse(a, b):
    return ((a - b)**2).mean()


def mae(a, b):
    return (abs(a - b)).mean()


def ssim_ref(img1, img2):
    # input: img1, img2, (H, W, C) uint8 image in [0, 255]
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim_ref_norm(img1, img2):
    # input: img1, img2, (H, W, C) normalized image in [0, 1]
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim_tensor(t1, t2):
    k1, k2 = 0.01, 0.03
    c1, c2 = k1**2, k2**2
    t1_mean = t1.mean((2, 3))[:, :, None, None]
    t2_mean = t2.mean((2, 3))[:, :, None, None]
    t1_sigma = torch.sqrt(((t1 - t1_mean)**2).mean((2, 3)))
    t2_sigma = torch.sqrt(((t2 - t2_mean)**2).mean((2, 3)))
    sigma_12 = ((t1 - t1_mean) * (t2 - t2_mean)).mean((2, 3))
    img_ssim = (2 * t1_mean * t2_mean + c1) * (2 * sigma_12 + c2) / (
        t1_mean**2 + t2_mean**2 + c1) / (t1_sigma**2 + t2_sigma**2 + c2)
    ssim = img_ssim.mean()
    return ssim

def psnr_tensor(t1, t2):
    Mse = ((t1 - t2)**2).mean((1, 2, 3))
    img_psnr = 10 * torch.log10(1. / (Mse + EPS))
    psnr = img_psnr.mean()
    return psnr

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

def pairwise_L2_dist(embedding, squared=False):
    '''
    input: (n, d) embeddings
    output: (n, n) (squared?) L2 distances
    '''
    dot_product = torch.mm(embedding, embedding.T)
    square_norm = torch.diag(dot_product)
    square_dist = torch.unsqueeze(square_norm, dim=1) - 2.0*dot_product + torch.unsqueeze(square_norm, dim=0)
    square_dist.clamp_min_(0.0)
    if not squared:
        dist = torch.sqrt(square_dist+1e-8)
        return dist
    return square_dist



class BatchHardTripletLoss(nn.Module):
    '''
    L2 triple loss
    input: (b, d) embeddings with class labels
    '''
    def __init__(self, batch_size, num_instances, margin=None):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None or self.margin==0:
            self.loss = nn.SoftMarginLoss()
        else:
            self.loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids = batch_size // num_instances
        self.labels = torch.arange(self.num_pids).repeat(num_instances, 1).T # [batch_size]
        self.labels = self.labels.contiguous().view(-1)
        self.positive_mask = self.get_anchor_positive_triplet_mask().type(torch.float)
        self.negative_mask = self.get_anchor_negative_triplet_mask().type(torch.float)

    def get_anchor_positive_triplet_mask(self):  
        indices_equal = torch.eye(self.labels.shape[0]).type(torch.bool)
        indices_not_equal = ~indices_equal
        labels_equal = torch.unsqueeze(self.labels, dim=0)==torch.unsqueeze(self.labels, dim=1)
        mask = indices_not_equal & labels_equal
        return mask

    def get_anchor_negative_triplet_mask(self):
        labels_equal = torch.unsqueeze(self.labels, dim=0)==torch.unsqueeze(self.labels, dim=1)
        mask = ~labels_equal
        return mask

    def forward(self, batch_embedding):
        dist = pairwise_L2_dist(batch_embedding, squared=False)
        self.positive_mask = self.positive_mask.to(batch_embedding.device)
        self.negative_mask = self.negative_mask.to(batch_embedding.device)
        anchor_postivie_dist = self.positive_mask * dist
        hardest_positive_dist = torch.max(anchor_postivie_dist, dim=1, keepdim=True)[0]
        max_anchor_dist = torch.max(dist, dim=1, keepdim=True)[0]
        anchor_negative_dist = dist + max_anchor_dist * (1 - self.negative_mask)
        hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]
        
        y = torch.ones((self.batch_size, 1))
        y = y.to(batch_embedding.device)
        if self.margin is None: 
            loss = self.loss(hardest_negative_dist - hardest_positive_dist, y)
        else:
            loss = self.loss(hardest_negative_dist, hardest_positive_dist, y)
        return loss


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    param x: [batch_size, hidden_size]
    param mu: [batch_size, hidden_size]
    pararm var: [batch_size, hidden_size]
    """
    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + EPS).log() - (x - mu).pow(2).div(var.mul(2.0) + EPS)
        nll = -(logli.sum(1).mean())

        return nll


class CosineSimilarity:
    def __call__(self, x1, x2):
        """
        Calculates the cosine similarity matrix for every pair (i, j),
        where i is an embedding from x1 and j is another embedding from x2.

        :param x1: a tensors with shape [batch_size, hidden_size].
        :param x2: a tensors with shape [batch_size, hidden_size].
        :return: the cosine similarity matrix with shape [batch_size, batch_size].
        """
        norm1 = torch.norm(x1, dim=1, keepdim=True)
        norm2 = torch.norm(x2, dim=1, keepdim=True)
        dot = torch.sum(x1*x2, dim=1, keepdim=True)
        
        return torch.mean(dot/(norm1*norm2+EPS))