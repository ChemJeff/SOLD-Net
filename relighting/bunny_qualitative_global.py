import os
import json
import argparse
import imageio
import numpy as np
from tqdm import tqdm

def mse(a, b):
    return ((a - b)**2).mean()


def mae(a, b):
    return (abs(a - b)).mean()

def calc_azimuth_error(pred, gt, unit='deg'):
    assert unit in ['deg', 'rad']
    cycle = 360 if unit == 'deg' else np.pi*2
    candidates = np.zeros(3)
    candidates[0] = pred - (gt - cycle)
    candidates[1] = pred - gt
    candidates[2] = pred - (gt + cycle)
    minabspos = np.argmin(abs(candidates))
    return candidates[minabspos]

def calc_angle_error(pred_az, pred_el, gt_az, gt_el, unit='deg'):
    assert unit.lower() in ['deg', 'rad']
    if unit=='deg':
        pred_az = pred_az / 180.0 * np.pi
        pred_el = pred_el / 180.0 * np.pi
        gt_az = gt_az / 180.0 * np.pi
        gt_el = gt_el / 180.0 * np.pi
    pred_unit = np.array([np.cos(pred_el)*np.sin(pred_az), # x
                            np.cos(pred_el)*np.cos(pred_az), # y
                            np.sin(pred_el)]) # z
    gt_unit = np.array([np.cos(gt_el)*np.sin(gt_az), # x
                            np.cos(gt_el)*np.cos(gt_az), # y
                            np.sin(gt_el)]) # z
    dot_product = np.einsum('i,i->', pred_unit, gt_unit)
    dot_product = np.clip(dot_product, a_min=-1.0, a_max=1.0)
    angle_error = np.arccos(dot_product)
    angle_error = angle_error / np.pi * 180.0
    angle_error = np.clip(angle_error, a_min=0.0, a_max=180.0)
    return angle_error

def rot_envmap_by_pixel(env, rot):
    env_rotate = np.zeros_like(env)
    # NOTE: if from left to right [0, 2pi]
    if rot<0: # rotate to left
        # rot = 63 - az
        rot = int(abs(rot))
        env_rotate[:,:-rot,:] = env[:,rot:,:]
        env_rotate[:,-rot:,:] = env[:,:rot,:]
    elif rot>0:
        # rot = az - 63 # rotate to right
        rot = int(abs(rot))
        env_rotate[:,rot:,:] = env[:,:-rot,:]
        env_rotate[:,:rot,:] = env[:,-rot:,:]
    else:
        env_rotate = env
    return env_rotate

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="./results/global/gt")
parser.add_argument("--mask_path", type=str, default="./mask/bunny_mask_global.png")
parser.add_argument("--splitfile", type=str, default="../data/synthetic/split/test.txt")
parser.add_argument("--result_dir", type=str, default="./results/global/sold")

args = parser.parse_args()

dataroot = args.dataroot
splitfile = args.splitfile
result_dir = args.result_dir
output_dir = result_dir

with open(splitfile, 'r') as f:
    items = [line.strip().split() for line in f.readlines() if line.strip()]

print("Found %s items" %(len(items)))

all_dict = {}
all_dict["all"] = {}
all_dict["stat"] = {}
gmse_list = []
gmae_list = []
grmse_list = []

bunny_mask = imageio.imread(args.mask_path)
_h, _w = bunny_mask.shape[0], bunny_mask.shape[1]
bunny_mask = bunny_mask.reshape(_h, _w, -1)
bunny_mask = bunny_mask[:,:,0]

for i, item in enumerate(tqdm(items)):
    city_cam_name, sky_name, angle_id = item
    filebasename = '%06d_%s_%s_%s'%(i+1, city_cam_name, sky_name, angle_id)
    est_global_path = os.path.join(result_dir, 'relight_bunny', '%s_bunny_relight.hdr' %(filebasename))
    gt_global_path = os.path.join(dataroot, 'relight_bunny', '%s_bunny_relight.hdr' %(filebasename))
    gt_global = imageio.imread(gt_global_path, format='HDR-FI')  # NOTE: No scale here for relighted bunnys!
    est_global = imageio.imread(est_global_path, format='HDR-FI') # H, W, C
    gt_global_nonzero = gt_global[bunny_mask>0]
    est_global_nonzero = est_global[bunny_mask>0]
    tmp_dict = {}
    if np.percentile(gt_global_nonzero, 10) > 1e-2:
        tmp_dict['GMAE'] = float(mae(gt_global_nonzero, est_global_nonzero))
        tmp_dict['GMSE'] = float(mse(gt_global_nonzero, est_global_nonzero))
        tmp_dict['GRMAE'] = float(np.sqrt(tmp_dict['GMSE']))
        gmae_list.append(tmp_dict['GMAE'])
        gmse_list.append(tmp_dict['GMSE'])
        grmse_list.append(tmp_dict['GRMAE'])
    else:
        tmp_dict['GMAE'] = -1.0
        tmp_dict['GMSE'] = -1.0
        tmp_dict['GRMAE'] = -1.0

    all_dict['all'][filebasename] = tmp_dict

gmse_array = np.array(gmse_list)
gmae_array = np.array(gmae_list)
grmse_array = np.array(grmse_list)

all_dict['stat']['AVG_GMAE'] = gmae_array.mean()
all_dict['stat']['AVG_GMSE'] = gmse_array.mean()
all_dict['stat']['AVG_GRMSE'] = grmse_array.mean()

print("avg_gmae: %f, avg_gmse: %f, avg_grmse: %f" %(all_dict['stat']['AVG_GMAE'], all_dict['stat']['AVG_GMSE'], all_dict['stat']['AVG_GRMSE']))

with open(os.path.join(output_dir, 'evaluation_global_relighting.json'), 'w') as f:
    f.write(json.dumps(all_dict, indent=2))

print("results have been saved to %s." %(os.path.join(output_dir, 'evaluation_global_relighting.json')))