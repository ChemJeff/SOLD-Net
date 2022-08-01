import os
import json
import argparse
import imageio
import numpy as np
from tqdm import tqdm
import glob

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

parser.add_argument("--dataroot", type=str, default="../../../data/real/")
parser.add_argument("--result_dir", type=str, default="../results/real/")
parser.add_argument("--output_dir", type=str, default="../results/real/")

args = parser.parse_args()

dataroot = args.dataroot
result_dir = args.result_dir
output_dir = args.output_dir

items = [os.path.split(path)[-1].split('_')[0] for path in sorted(glob.glob(os.path.join(dataroot, '*_persp.*')))]

print("Found %s items" %(len(items)))

all_dict = {}
all_dict["all"] = {}
all_dict["stat"] = {}
mse_list = []
mae_list = []
rmse_list = []
az = []
el = []
deg_error_list = []

for i, item in enumerate(tqdm(items)):
    meta_idx = item
    filebasename = '%s'%(meta_idx)
    est_global_path = os.path.join(result_dir, 'dump', '%s_global_est.hdr' %(filebasename))
    est_global = imageio.imread(est_global_path, format='HDR-FI') # H, W, C

    sun_pos_path = os.path.join(dataroot, '%s_sun_pos.txt' %(meta_idx))
    sun_pos = np.loadtxt(sun_pos_path)

    azimuth_deg_gt = (sun_pos[0] - 63.5) / 64 * 180
    elevation_deg_gt = (31.5 - sun_pos[1]) / 64 * 180

    est_int = np.sum(est_global, axis=2)
    H, W = est_int.shape
    assert(H==32)
    assert(W==128)
    sun_pos_est = np.unravel_index(np.argmax(est_int), est_int.shape)
    est_azimuth = (sun_pos_est[1] - 63.5) / 64 * 180
    est_elevation = (31.5 - sun_pos_est[0]) / 64 * 180

    rot_ind = -int(np.around(float(calc_azimuth_error(est_azimuth, azimuth_deg_gt, unit='deg')) / 180.0 * 64.0, 0)) # - to left, + to right

    tmp_dict = all_dict['all'].get(meta_idx, {})

    for local_idx in range(2):
        subdirname = '_'.join([meta_idx, str(local_idx)])
        gt_local_path = os.path.join(dataroot, '%s_local_%s.hdr' %(meta_idx, str(local_idx)))
        est_local_path = os.path.join(result_dir, 'dump', '%s_local_%s_est.hdr' %(meta_idx, str(local_idx)))
        gt_local = imageio.imread(gt_local_path, format='HDR-FI')
        gt_local[gt_local>2000.0] = 2000.0 # clip for outliers
        est_local = imageio.imread(est_local_path, format='HDR-FI') # H, W, C

        _tmp_dict = {}
        _tmp_dict['MAE'] = float(mae(est_local, gt_local))
        _tmp_dict['MSE'] = float(mse(est_local, gt_local))
        _tmp_dict['RMSE'] = float(np.sqrt(_tmp_dict['MSE']))
        tmp_dict[local_idx] = _tmp_dict
        mae_list.append(_tmp_dict['MAE'])
        mse_list.append(_tmp_dict['MSE'])
        rmse_list.append(_tmp_dict['RMSE'])

    tmp_dict['az'] = float(calc_azimuth_error(est_azimuth, azimuth_deg_gt, unit='deg'))
    tmp_dict['el'] = float(est_elevation - elevation_deg_gt)
    tmp_dict['deg_error'] = float(calc_angle_error(est_azimuth, est_elevation, azimuth_deg_gt, elevation_deg_gt, unit='deg'))
    az.append(tmp_dict['az'])
    el.append(tmp_dict['el'])
    deg_error_list.append(tmp_dict['deg_error'])

    all_dict['all'][meta_idx] = tmp_dict

mae_array = np.array(mae_list)
mse_array = np.array(mse_list)
rmse_array = np.array(rmse_list)
az_array = np.array(az)
el_array = np.array(el)
deg_error_array = np.array(deg_error_list)
all_dict['stat']['AVG_MAE'] = mae_array.mean()
all_dict['stat']['AVG_MSE'] = mse_array.mean()
all_dict['stat']['AVG_RMSE'] = rmse_array.mean()
all_dict['stat']['AVG_AZ_MAE'] = abs(az_array).mean()
all_dict['stat']['AVG_EL_MAE'] = abs(el_array).mean()
all_dict['stat']['AVG_DEG_ERR'] = deg_error_array.mean()
print("avg_mae: %f, avg_mse: %f, avg_rmse: %f" %(all_dict['stat']['AVG_MAE'], all_dict['stat']['AVG_MSE'], all_dict['stat']['AVG_RMSE']))
print("avg_az_mae: %f, avg_el_mae: %f" %(all_dict['stat']['AVG_AZ_MAE'], all_dict['stat']['AVG_EL_MAE']))
print("avg_deg_err: %f" %(all_dict['stat']['AVG_DEG_ERR']))

with open(os.path.join(output_dir, 'evaluation_local.json'), 'w') as f:
    f.write(json.dumps(all_dict, indent=2))
