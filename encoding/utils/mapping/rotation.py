import torch

def RotateByPixel(input_img, rotateX_pixels):
    if rotateX_pixels != 0:
        rotated_img = torch.zeros_like(input_img, device=input_img.device)
        rotated_img[:,:,:,:-rotateX_pixels] = input_img[:,:,:,rotateX_pixels:]
        rotated_img[:,:,:,-rotateX_pixels:] = input_img[:,:,:,:rotateX_pixels]
    else:
        rotated_img = input_img
    return rotated_img