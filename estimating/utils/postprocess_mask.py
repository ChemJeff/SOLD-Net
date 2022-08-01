import cv2
import numpy as np

dilation_kernel = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]], dtype=np.uint8
)

erosion_kernel = np.array(
    [[0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [0, 1, 1, 1, 0]], dtype=np.uint8
)

def largest_connected_region(binary_mask):
    # binary_mask: H, W
    # _, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = []
    for j in range(len(contours)):
        areas.append(cv2.contourArea(contours[j]))
    
    max_idx = np.argmax(areas)

    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(binary_mask, [contours[k]], 0)
    
    return binary_mask

def pred_sun_vis(input_shadow_mask, input_pos, shadow_thres=0.75, sun_thres=0.25):
    # input_shadow_mask: 1, H, W; input_pos: 2
    # select 7*7 patch of shadow mask
    _, H, W = input_shadow_mask.shape
    idx_y = input_pos[0]
    idx_x = input_pos[1]
    idx_up = np.maximum(0, idx_y-3)
    idx_down = np.minimum(H, idx_up+8)
    idx_up = idx_down - 8
    idx_left = np.maximum(0, idx_x-3)
    idx_right = np.minimum(W, idx_left+8)
    idx_left = idx_right - 8
    shadow_patch = input_shadow_mask[0,idx_up:idx_down,idx_left:idx_right]
    # check thresholds
    if np.all(shadow_patch>shadow_thres):
        return "shadowed"
    elif np.all(shadow_patch<sun_thres):
        return "non-shadowed"
    else:
        return "not-sure"


def postprocess_sun_pos(input_sun_pos, input_sun_pano=None, sun_thres=1.0):
    # input_sun_pos: 1, H, W; input_sun_pano: 3, H, W
    # dilation and blur
    if input_sun_pano is not None:
        sun_pos_mask = np.uint8(input_sun_pano.mean(axis=0, keepdims=True)>sun_thres)
        cv2.dilate(sun_pos_mask, dilation_kernel, iterations=1)
        cv2.erode(sun_pos_mask, erosion_kernel, iterations=1)
    else:
        sun_pos_mask = input_sun_pos
    sun_pos_mask = cv2.GaussianBlur(sun_pos_mask.astype(np.float32), (5, 5), sigmaX=0)
    
    return sun_pos_mask

def postprocess_local_sil(input_local_sil, input_sun_pos=None, sil_thres=0.7):
    # input_local_sil: 1, H, W; input_sun_pos: 1, H, W
    if input_sun_pos is not None:
        input_local_sil[:,:32,:] = input_local_sil[:,:32,:] - input_sun_pos
    sil_pos_mask = np.uint8(input_local_sil > sil_thres)
    # find largest connected area
    lca = largest_connected_region(sil_pos_mask[0])
    sil_pos_mask[0] = lca
    sil_pos_mask = sil_pos_mask.astype(np.uint8)

    # erosion and blur
    cv2.erode(sil_pos_mask, erosion_kernel, iterations=1)
    sil_pos_mask = sil_pos_mask.astype(np.float32)
    sil_pos_mask[0,31:,:] = 1.0 # NOTE: lower semisphere always true!
    sil_pos_mask = cv2.GaussianBlur(sil_pos_mask, (5, 5), sigmaX=0)

    return sil_pos_mask