import argparse
import os
import pickle
import json
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

parser.add_argument("--dataroot", type=str, default="../../../data/synthetic")
parser.add_argument("--splitfile", type=str, default="../../../data/synthetic/split/test.txt")
parser.add_argument("--result_dir", type=str, default="./results/synthetic/")
parser.add_argument("--output_dir", type=str, default="./results/synthetic/")

args = parser.parse_args()

dataroot = args.dataroot
splitfile = args.splitfile
result_dir = args.result_dir
output_dir = args.output_dir

with open(splitfile, 'r') as f:
    items = [line.strip().split() for line in f.readlines() if line.strip()]

print("Found %s items" %(len(items)))

all_dict = {}
all_dict["all"] = {}
all_dict["stat"] = {}
gmse_list = []
gmae_list = []
grmse_list = []
gmse_n_list = []
gmae_n_list = []
grmse_n_list = []  
az = []
el = []
deg_error_list = []

for i, item in enumerate(tqdm(items)):
    city_cam_name, sky_name, angle_id = item
    filebasename = '%06d_%s_%s_%s'%(i+1, city_cam_name, sky_name, angle_id)
    est_global_path = os.path.join(result_dir, 'dump', '%s_global_est.hdr' %(filebasename))
    gt_global_path = os.path.join(dataroot, city_cam_name, 'rendered', sky_name, 'global_lighting_%s.hdr' %(angle_id))
    gt_sun_stat_path = os.path.join(dataroot, city_cam_name, 'rendered', sky_name, 'sun_pos_%s.pkl' %(angle_id))
    with open(gt_sun_stat_path, 'rb') as f:
        sun_stat = pickle.load(f)
    gt_global = imageio.imread(gt_global_path, format='HDR-FI')*4.0 # NOTE: fix scale
    est_global = imageio.imread(est_global_path, format='HDR-FI') # H, W, C

    tmp_dict = {}
    tmp_dict['GMAE'] = float(mae(gt_global, est_global))
    tmp_dict['GMSE'] = float(mse(gt_global, est_global))
    tmp_dict['GRMAE'] = float(np.sqrt(tmp_dict['GMSE']))
    gmae_list.append(tmp_dict['GMAE'])
    gmse_list.append(tmp_dict['GMSE'])
    grmse_list.append(tmp_dict['GRMAE'])
    sun_azimuth = sun_stat['azimuth']
    sun_elevation = sun_stat['elevation']
    sun_vis = sun_stat['is_sunny']
    azimuth_deg_gt = sun_azimuth / np.pi * 180
    elevation_deg_gt = sun_elevation / np.pi * 180
    est_int = np.sum(est_global, axis=2)
    H, W = est_int.shape
    assert(H==32)
    assert(W==128)
    sun_pos_est = np.unravel_index(np.argmax(est_int), est_int.shape)
    est_azimuth = (sun_pos_est[1] - 63.5) / 64 * 180
    est_elevation = (31.5 - sun_pos_est[0]) / 64 * 180

    rot_ind = -int(np.around(float(calc_azimuth_error(est_azimuth, azimuth_deg_gt, unit='deg')) / 180.0 * 64.0, 0)) # - to left, + to right

    est_global_norm = rot_envmap_by_pixel(est_global, rot_ind)
    tmp_dict['GMAE_N'] = float(mae(gt_global, est_global_norm))
    tmp_dict['GMSE_N'] = float(mse(gt_global, est_global_norm))
    tmp_dict['GRMAE_N'] = float(np.sqrt(tmp_dict['GMSE_N']))
    gmae_n_list.append(tmp_dict['GMAE_N'])
    gmse_n_list.append(tmp_dict['GMSE_N'])
    grmse_n_list.append(tmp_dict['GRMAE_N'])

    if sun_vis == 1:
        tmp_dict['az'] = float(calc_azimuth_error(est_azimuth, azimuth_deg_gt, unit='deg'))
        tmp_dict['el'] = float(est_elevation - elevation_deg_gt)
        tmp_dict['deg_error'] = float(calc_angle_error(est_azimuth, est_elevation, azimuth_deg_gt, elevation_deg_gt, unit='deg'))
        az.append(tmp_dict['az'])
        el.append(tmp_dict['el'])
        deg_error_list.append(tmp_dict['deg_error'])

    all_dict['all'][filebasename] = tmp_dict

gmse_array = np.array(gmse_list)
gmae_array = np.array(gmae_list)
grmse_array = np.array(grmse_list)
gmse_n_array = np.array(gmse_n_list)
gmae_n_array = np.array(gmae_n_list)
grmse_n_array = np.array(grmse_n_list)
az_array = np.array(az)
el_array = np.array(el)
deg_error_array = np.array(deg_error_list)
all_dict['stat']['AVG_GMAE'] = gmae_array.mean()
all_dict['stat']['AVG_GMSE'] = gmse_array.mean()
all_dict['stat']['AVG_GRMSE'] = grmse_array.mean()
all_dict['stat']['AVG_GMAE_N'] = gmae_n_array.mean()
all_dict['stat']['AVG_GMSE_N'] = gmse_n_array.mean()
all_dict['stat']['AVG_GRMSE_N'] = grmse_n_array.mean()
all_dict['stat']['AVG_AZ_MAE'] = abs(az_array).mean()
all_dict['stat']['AVG_EL_MAE'] = abs(el_array).mean()
all_dict['stat']['AVG_DEG_ERR'] = deg_error_array.mean()
print("avg_gmae: %f, avg_gmse: %f, avg_grmse: %f" %(all_dict['stat']['AVG_GMAE'], all_dict['stat']['AVG_GMSE'], all_dict['stat']['AVG_GRMSE']))
print("avg_gmae_n: %f, avg_gmse_n: %f, avg_grmse_n: %f" %(all_dict['stat']['AVG_GMAE_N'], all_dict['stat']['AVG_GMSE_N'], all_dict['stat']['AVG_GRMSE_N']))
print("avg_az_mae: %f, avg_el_mae: %f" %(all_dict['stat']['AVG_AZ_MAE'], all_dict['stat']['AVG_EL_MAE']))
print("avg_deg_err: %f" %(all_dict['stat']['AVG_DEG_ERR']))

with open(os.path.join(output_dir, 'evaluation_global.json'), 'w') as f:
    f.write(json.dumps(all_dict, indent=2))
