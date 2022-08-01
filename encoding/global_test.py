import os
import argparse
import json
import torch
import imageio
import numpy as np
from data.dataset_laval_sky import LavalSkyDataset
from data.dataset_synthetic_global import SynGlobalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss.pytorch_ssim import ssim
from utils.mapping.log_mapping import linear2log, log2linear
from utils.tonemapping import GammaTMO
from utils.loss import mae, mse
from utils.logger import *

from model.Autoencoder import GlobalEncoder, SkyDecoder, SunDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true")
parser.add_argument("--dataset", type=str, choices=['laval', 'synthetic'], default='laval')
parser.add_argument("--dataroot_laval", type=str, default='../data/envmaps/LavalSkyHDR/128')
parser.add_argument("--dataroot_sun_laval", type=str, default='../data/envmaps/LavalSkyHDR/stats/128')
parser.add_argument("--dataroot_syn", type=str, default="../data/synthetic")
parser.add_argument("--sunmask_scale", type=int, default=4)
parser.add_argument("--sunmask_patchsize", type=int, default=8)
parser.add_argument("--split_dir_laval", type=str, default='./data/split_laval')
parser.add_argument("--split_dir_syn", type=str, default='../data/synthetic/split')
parser.add_argument("--filterfile_laval", type=str, nargs='+', default=None)
parser.add_argument("--filterfile_syn", type=str, nargs='+', default=None)
parser.add_argument("--log_image", action="store_true", help="use image in log space")
parser.add_argument("--log_mu", type=float, default=16.0)
parser.add_argument("--load_sky_enc_path", type=str, required=True)
parser.add_argument("--load_sun_enc_path", type=str, required=True)
parser.add_argument("--load_sky_dec_path", type=str, required=True)
parser.add_argument("--load_sun_dec_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--sky_dim", type=int, default=16)
parser.add_argument("--sun_dim", type=int, default=45)
parser.add_argument("--tmo_gamma", type=float, default=2.2)
parser.add_argument("--tmo_log_exposure", type=float, default=-2)
parser.add_argument("--plot_every", type=int, required=True)
parser.add_argument("--num_loader", type=int, default=8)
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--test_split", type=str, default='test')
parser.add_argument("--model_activ", type=str, choices=['relu', 'lrelu'], default='relu')

args = parser.parse_args()

if args.dataset == 'laval':
    dataroot = args.dataroot_laval
    testsplit = os.path.join(args.split_dir_laval, args.test_split+'.txt')
elif args.dataset == 'synthetic':
    dataroot = args.dataroot_syn
    testsplit = os.path.join(args.split_dir_syn, args.test_split+'.txt')

os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_dir), exist_ok=True)
if args.test_split != "test":
    output_dir = os.path.join(args.result_dir, args.test_split, args.dataset)
else:
    output_dir = os.path.join(args.result_dir, args.dataset)
os.makedirs(output_dir, exist_ok=(args.debug or args.override))
output_image_dir = os.path.join(output_dir, 'image')
os.makedirs(output_image_dir, exist_ok=(args.debug or args.override))
output_image_ldr_dir = os.path.join(output_dir, 'image_ldr')
os.makedirs(output_image_ldr_dir, exist_ok=(args.debug or args.override))

save_options_cmdline(output_dir, args)
logger = set_logger(output_dir)
tb_logger = set_tb_logger(output_dir)
tb_save_options_cmdline(tb_logger, args)

# initialize models
enc_sky = GlobalEncoder(cin=3, cout=args.sky_dim, activ=args.model_activ).to('cuda')
enc_sun = GlobalEncoder(cin=3, cout=args.sun_dim, activ=args.model_activ).to('cuda')
dec_sky = SkyDecoder(cin=args.sky_dim, cout=3, activ=args.model_activ).to('cuda')
dec_sun = SunDecoder(cin=args.sun_dim, cout=3, activ=args.model_activ).to('cuda')
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
BCE = torch.nn.BCELoss(reduction='mean')

print("output path: ", output_dir)

# load checkpoints
print('loading sky encoder from ', args.load_sky_enc_path)
enc_sky.load_state_dict(torch.load(args.load_sky_enc_path, map_location='cuda'))
print('loading sky decoder from ', args.load_sky_dec_path)
dec_sky.load_state_dict(torch.load(args.load_sky_dec_path, map_location='cuda'))
print('loading sun encoder from ', args.load_sun_enc_path)
enc_sun.load_state_dict(torch.load(args.load_sun_enc_path, map_location='cuda'))
print('loading sun decoder from ', args.load_sun_dec_path)
dec_sun.load_state_dict(torch.load(args.load_sun_dec_path, map_location='cuda'))

if args.dataset == 'laval':
    testSet = LavalSkyDataset(opt=args, dataroot=dataroot, splitfile=testsplit, phase=args.test_split, filterfile=args.filterfile_laval, dataroot_sun=args.dataroot_sun_laval)
elif args.dataset == 'synthetic':
    testSet = SynGlobalDataset(opt=args, dataroot=dataroot, splitfile=testsplit, phase=args.test_split, filterfile=args.filterfile_syn)
testLoader = DataLoader(testSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_loader)

enc_sky.eval()
dec_sky.eval()
enc_sun.eval()
dec_sun.eval()

all_dict = {}
all_dict["all"] = {}
all_dict["stat"] = {}
mse_list = []
mae_list = []
rmse_list = []
ldr_mse_list = []
ldr_mae_list = []
ldr_ssim_list = []

image_cnt = 0

with torch.no_grad():
    num_vis = 0
    for i, test_data in tqdm(enumerate(testLoader)):
        image_tensor = test_data['color'].to('cuda')
        if args.dataset == 'synthetic':
            image_tensor = image_tensor*4.0
        image_tensor.clamp_min_(0.0)
        image_tensor.clamp_max_(2000.0)
        sun_pos_mask_fine = test_data['sun_pos_mask_fine'].to('cuda')
        if args.dataset == 'laval':
            env_date = test_data['date']
            env_time = test_data['time']
        elif args.dataset == 'synthetic':
            city_cam_name = test_data['city_cam_name']
            sky_name = test_data['sky_name']
            angle_id = test_data['angle_id']

        image_target_tensor = image_tensor
        image_target_sky_tensor = image_target_tensor.clamp_max(1.0)

        if args.log_image:
            input_tensor = recon_target_tensor = linear2log(image_target_tensor, args.log_mu)
            input_sky_tensor = recon_target_sky_tensor = linear2log(image_target_sky_tensor, args.log_mu)
        else:
            input_tensor = recon_target_tensor = image_target_tensor
            input_sky_tensor = recon_target_sky_tensor = image_target_sky_tensor

        latent_sky = enc_sky(input_sky_tensor)
        latent_sun = enc_sun(input_tensor)

        recon_sky_tensor = dec_sky(latent_sky)
        recon_sun_tensor = dec_sun(latent_sun, sun_pos_mask_fine)

        if args.log_image:
            image_recon_sky_tensor = log2linear(recon_sky_tensor.clamp_min(0.0).clamp_max(4.5), args.log_mu)
            image_recon_sun_tensor = log2linear(recon_sun_tensor.clamp_min(0.0).clamp_max(4.5), args.log_mu)
            image_recon_tensor = image_recon_sky_tensor*(1-sun_pos_mask_fine)+ image_recon_sun_tensor
            # image_recon_tensor = image_recon_sky_tensor+ image_recon_sun_tensor # NOTE: do not use mask here for more visually pleasing results
        else:
            image_recon_sky_tensor = recon_sky_tensor.clamp_min(0.0)
            image_recon_sun_tensor = recon_sun_tensor.clamp_min(0.0)
            image_recon_tensor = image_recon_sky_tensor*(1-sun_pos_mask_fine)+ image_recon_sun_tensor
            # image_recon_tensor = image_recon_sky_tensor+ image_recon_sun_tensor

        for k in range(image_recon_tensor.shape[0]):
            img_idx = image_cnt + k
            if args.dataset == 'laval':
                filebasename = '%06d_%s_%s'%(img_idx, env_date[k], env_time[k])
            elif args.dataset == 'synthetic':
                filebasename = '%06d_%s_%s_%s'%(img_idx, city_cam_name[k], sky_name[k], angle_id[k])

            gt_image = image_tensor[k:k+1]
            recon_image = image_recon_tensor[k:k+1]
            recon_sky_image = image_recon_sky_tensor[k:k+1]
            recon_sun_image = image_recon_sun_tensor[k:k+1]

            gt_ldr = GammaTMO(gt_image, args.tmo_gamma, args.tmo_log_exposure) # 1, C, H, W
            recon_ldr = GammaTMO(recon_image, args.tmo_gamma, args.tmo_log_exposure) # 1, C, H, W
            recon_sky_ldr = GammaTMO(recon_sky_image, args.tmo_gamma, args.tmo_log_exposure) # 1, C, H, W
            recon_sun_ldr = GammaTMO(recon_sun_image, args.tmo_gamma, args.tmo_log_exposure) # 1, C, H, W

            tmp_dict = {}
            tmp_dict['MAE'] = mae(gt_image, recon_image).cpu().item()
            tmp_dict['MSE'] = mse(gt_image, recon_image).cpu().item()
            tmp_dict['RMSE'] = np.sqrt(tmp_dict['MSE'])
            tmp_dict['LDR_MAE'] = mae(gt_ldr, recon_ldr).cpu().item()
            tmp_dict['LDR_MSE'] = mse(gt_ldr, recon_ldr).cpu().item()
            tmp_dict['LDR_SSIM'] = ssim(gt_ldr, recon_ldr).cpu().item()
            all_dict["all"][filebasename] = tmp_dict
            mae_list.append(tmp_dict['MAE'])
            mse_list.append(tmp_dict['MSE'])
            rmse_list.append(tmp_dict['RMSE'])
            ldr_mae_list.append(tmp_dict['LDR_MAE'])
            ldr_mse_list.append(tmp_dict['LDR_MSE'])
            ldr_ssim_list.append(tmp_dict['LDR_SSIM'])

            input_img_path = os.path.join(output_image_dir, '%s_input.hdr' %(filebasename))
            _input_img_tensor = np.transpose(gt_image[0].cpu(), (1, 2, 0)) # H * W * C
            
            recon_img_path = os.path.join(output_image_dir, '%s_recon.hdr' %(filebasename))
            _recon_img_tensor = np.transpose(recon_image[0].cpu(), (1, 2, 0)) # H * W * C
            
            gt_sun_pos_path = os.path.join(output_image_dir, '%s_sun_pos_gt.png' %(filebasename))
            gt_sun_pos_ldr_path = os.path.join(output_image_ldr_dir, '%s_sun_pos_gt.png' %(filebasename))
            _gt_sun_pos_tensor = (np.transpose(sun_pos_mask_fine[k].cpu().numpy(), (1, 2, 0)) * 255.0).astype(np.uint8) # H * W * C
            _sun_pos_mask_sample = np.repeat(_gt_sun_pos_tensor, 3, axis=2)
            
            recon_sky_path = os.path.join(output_image_dir, '%s_recon_sky.hdr' %(filebasename))
            _recon_sky_tensor = np.transpose(recon_sky_image[0].cpu(), (1, 2, 0)) # H * W * C
            
            recon_sun_path = os.path.join(output_image_dir, '%s_recon_sun.hdr' %(filebasename))
            _recon_sun_tensor = np.transpose(recon_sun_image[0].cpu(), (1, 2, 0)) # H * W * C
            
            input_img_ldr_path = os.path.join(output_image_ldr_dir, '%s_input.png' %(filebasename))
            _input_img_ldr_tensor = (np.transpose(gt_ldr[0].cpu().numpy(), (1, 2, 0)) * 255.0).astype(np.uint8) # H * W * C
            
            recon_img_ldr_path = os.path.join(output_image_ldr_dir, '%s_recon.png' %(filebasename))
            _recon_img_ldr_tensor = (np.transpose(recon_ldr[0].cpu().numpy(), (1, 2, 0)) * 255.0).astype(np.uint8) # H * W * C

            recon_sky_ldr_path = os.path.join(output_image_ldr_dir, '%s_recon_sky.png' %(filebasename))
            _recon_sky_ldr_tensor = (np.transpose(recon_sky_ldr[0].cpu().numpy(), (1, 2, 0)) * 255.0).astype(np.uint8) # H * W * C

            recon_sun_ldr_path = os.path.join(output_image_ldr_dir, '%s_recon_sun.png' %(filebasename))
            _recon_sun_ldr_tensor = (np.transpose(recon_sun_ldr[0].cpu().numpy(), (1, 2, 0)) * 255.0).astype(np.uint8) # H * W * C

            # imageio.imwrite(input_img_path, _input_img_tensor, format='HDR-FI')
            # imageio.imwrite(recon_img_path, _recon_img_tensor, format='HDR-FI')
            # imageio.imwrite(recon_sky_path, _recon_sky_tensor, format='HDR-FI')
            # imageio.imwrite(recon_sun_path, _recon_sun_tensor, format='HDR-FI')
            # imageio.imwrite(gt_sun_pos_path, _gt_sun_pos_tensor)
            # imageio.imwrite(input_img_ldr_path, _input_img_ldr_tensor)
            # imageio.imwrite(recon_img_ldr_path, _recon_img_ldr_tensor)
            # imageio.imwrite(recon_sky_ldr_path, _recon_sky_ldr_tensor)
            # imageio.imwrite(recon_sun_ldr_path, _recon_sun_ldr_tensor)
            # imageio.imwrite(gt_sun_pos_ldr_path, _gt_sun_pos_tensor)
            
            # NOTE: reduce disk I/O ops by merging images as subplots
            image_sample_path = os.path.join(output_image_dir, '%s_sample.hdr' %(filebasename))
            image_sample = np.concatenate([_input_img_tensor, _recon_img_tensor, _sun_pos_mask_sample.astype(np.float32)],  axis=1)
            image_split_sample = np.concatenate([_recon_sky_tensor, _recon_sun_tensor, _sun_pos_mask_sample.astype(np.float32)], axis=1)
            image_sample = np.concatenate([image_sample, image_split_sample], axis=0)
            imageio.imwrite(image_sample_path, image_sample)

            image_ldr_sample_path = os.path.join(output_image_ldr_dir, '%s_sample.png' %(filebasename))
            image_ldr_sample = np.concatenate([_input_img_ldr_tensor, _recon_img_ldr_tensor, _sun_pos_mask_sample],  axis=1)
            image_ldr_split_sample = np.concatenate([_recon_sky_ldr_tensor, _recon_sun_ldr_tensor, _sun_pos_mask_sample], axis=1)
            image_ldr_sample = np.concatenate([image_ldr_sample, image_ldr_split_sample], axis=0)
            imageio.imwrite(image_ldr_sample_path, image_ldr_sample)
        
            if tb_logger and (i+1) % args.plot_every == 0:
                add_tf_summary_value(tb_logger, "MAE", tmp_dict['MAE'], iteration=img_idx)
                add_tf_summary_value(tb_logger, "MSE", tmp_dict['MSE'], iteration=img_idx)
                add_tf_summary_value(tb_logger, "RMSE", tmp_dict['RMSE'], iteration=img_idx)
                add_tf_summary_value(tb_logger, "LDR_MAE", tmp_dict['LDR_MAE'], iteration=img_idx)
                add_tf_summary_value(tb_logger, "LDR_MSE", tmp_dict['LDR_MSE'], iteration=img_idx)
                add_tf_summary_value(tb_logger, "LDR_SSIM", tmp_dict['LDR_SSIM'], iteration=img_idx)

        image_cnt += image_recon_tensor.shape[0]

avg_mae = np.array(mae_list, dtype=np.float64).mean()
avg_mse = np.array(mse_list, dtype=np.float64).mean()
avg_rmse = np.array(rmse_list, dtype=np.float64).mean()
avg_ldr_mae = np.array(ldr_mae_list, dtype=np.float64).mean()
avg_ldr_mse = np.array(ldr_mse_list, dtype=np.float64).mean()
avg_ldr_ssim = np.array(ldr_ssim_list, dtype=np.float64).mean()
all_dict['stat']['AVG_MAE'] = avg_mae
all_dict['stat']['AVG_MSE'] = avg_mse
all_dict['stat']['AVG_RMSE'] = avg_rmse
all_dict['stat']['AVG_LDR_MAE'] = avg_ldr_mae
all_dict['stat']['AVG_LDR_MSE'] = avg_ldr_mse
all_dict['stat']['AVG_LDR_SSIM'] = avg_ldr_ssim
print("avg_mae: %f, avg_mse: %f, avg_rmse: %f" %(avg_mae, avg_mse, avg_rmse))
print("avg_ldr_mae: %f, avg_ldr_mse: %f" %(avg_ldr_mae, avg_ldr_mse))
print("avg_ldr_ssim: %f" %(avg_ldr_ssim))
with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
    f.write(json.dumps(all_dict, indent=2))

print()
print("All done. Results have been saved to %s" %(output_dir))
