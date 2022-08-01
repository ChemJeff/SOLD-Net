import os
import argparse
import json
import imageio
import torch
import random
import numpy as np
from data.dataset_synthetic_local import SynLocalDataset
from data.local_identity_sampler import RandomLocalIdentiyiSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Autoencoder import GlobalEncoder, SkyDecoder, SunDecoder
from model.Autoencoder import LocalEncoder, LocalSilDecoder, LocalAppSplitRenderer
from utils.tonemapping import GammaTMO
from utils.loss import mae, mse
from utils.loss.pytorch_ssim import ssim
from utils.loss import CosineSimilarity, NormalNLLLoss
from utils.mapping.log_mapping import linear2log, log2linear
from utils.logger import *

random_seed = 1998
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true")
parser.add_argument("--dataroot_syn", type=str, default="../data/synthetic")
parser.add_argument("--split_dir_syn", type=str, default='../data/synthetic/split')
parser.add_argument("--filterfile_syn", type=str, nargs='+', default=None)
parser.add_argument("--log_image", action="store_true", help="use image in log space")
parser.add_argument("--log_mu", type=float, default=16.0)
parser.add_argument("--load_local_enc_path", type=str, required=True)
parser.add_argument("--load_app_dec_path", type=str, required=True)
parser.add_argument("--load_sil_dec_path", type=str, required=True)
parser.add_argument("--load_sky_enc_path", type=str, required=True)
parser.add_argument("--load_sun_enc_path", type=str, required=True)
parser.add_argument("--load_sky_dec_path", type=str, required=True)
parser.add_argument("--load_sun_dec_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--sky_dim", type=int, default=16)
parser.add_argument("--sun_dim", type=int, default=45)
parser.add_argument("--tmo_gamma", type=float, default=2.2)
parser.add_argument("--tmo_log_exposure", type=float, default=1)
parser.add_argument("--num_loader", type=int, default=1)
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--test_split", type=str, default='test')
parser.add_argument("--model_activ", type=str, choices=['relu', 'lrelu'], default='relu')
parser.add_argument("--cosine_random", action="store_true")
parser.add_argument("--cosine_zero", action="store_true")
parser.add_argument("--cross_render", action="store_true")

args = parser.parse_args()

assert(args.batch_size%2 == 0) # NOTE: must be pairwise data

dataroot_syn = args.dataroot_syn
testsplit_syn = os.path.join(args.split_dir_syn, args.test_split+'.txt')

os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_dir), exist_ok=True)

assert(not (args.cosine_random and args.cosine_zero))

if args.cosine_random:
    output_subdir = os.path.join('noise_cosine')
elif args.cosine_zero:
    output_subdir = os.path.join('zero_cosine')
else:
    output_subdir = ''

if args.cross_render:
    output_subdir = os.path.join('quantitative', 'cross_render', output_subdir)
else:
    output_subdir = os.path.join('quantitative', output_subdir)

if args.test_split != "test":
    output_dir = os.path.join(args.result_dir, args.test_split, output_subdir)
else:
    output_dir = os.path.join(args.result_dir, output_subdir)
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
enc_global_sky = GlobalEncoder(cin=3, cout=args.sky_dim, activ=args.model_activ).to('cuda')
print('loading sky encoder from ', args.load_sky_enc_path)
enc_global_sky.load_state_dict(torch.load(args.load_sky_enc_path, map_location='cuda'))
enc_global_sky.eval()
enc_global_sun = GlobalEncoder(cin=3, cout=args.sun_dim, activ=args.model_activ).to('cuda')
print('loading sun encoder from ', args.load_sun_enc_path)
enc_global_sun.load_state_dict(torch.load(args.load_sun_enc_path, map_location='cuda'))
enc_global_sun.eval()
dec_global_sky = SkyDecoder(cin=args.sky_dim, cout=3, activ=args.model_activ).to('cuda')
print('loading sky decoder from ', args.load_sky_dec_path)
dec_global_sky.load_state_dict(torch.load(args.load_sky_dec_path, map_location='cuda'))
dec_global_sky.eval()
dec_global_sun = SunDecoder(cin=args.sun_dim, cout=3, activ=args.model_activ).to('cuda')
print('loading sun decoder from ', args.load_sun_dec_path)
dec_global_sun.load_state_dict(torch.load(args.load_sun_dec_path, map_location='cuda'))
dec_global_sun.eval()
enc_local = LocalEncoder(cin=3, cout=64, activ=args.model_activ).to('cuda')
dec_sil = LocalSilDecoder(cin=64, cout=1, activ=args.model_activ).to('cuda')
dec_app = LocalAppSplitRenderer(cin_l=64, cin_sky=args.sky_dim, cin_sun=args.sun_dim, cout=3, activ=args.model_activ).to('cuda')
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
BCE = torch.nn.BCELoss(reduction='mean')
CE = torch.nn.CrossEntropyLoss(reduction='mean')
COS = CosineSimilarity()
NNLL = NormalNLLLoss()

print("output path: ", output_dir)

# load checkpoints
print('loading local encoder from ', args.load_local_enc_path)
enc_local.load_state_dict(torch.load(args.load_local_enc_path, map_location='cuda'))
print('loading silhouette decoder from ', args.load_sil_dec_path)
dec_sil.load_state_dict(torch.load(args.load_sil_dec_path, map_location='cuda'))
print('loading appearance decoder from ', args.load_app_dec_path)
dec_app.load_state_dict(torch.load(args.load_app_dec_path, map_location='cuda'))

testSet = SynLocalDataset(opt=args, dataroot=dataroot_syn, splitfile=testsplit_syn, phase=args.test_split, filterfile=args.filterfile_syn)

testSampler = RandomLocalIdentiyiSampler(testSet, dataroot=dataroot_syn, batch_size=args.batch_size, num_instance=2) # a pair of same local different global for cross-render constraint

testLoader = DataLoader(testSet, batch_size=args.batch_size, sampler=testSampler, num_workers=args.num_loader, drop_last=True) # NOTE: 'shuffle' option inactive here, and 'drop_last' must be set True!

enc_global_sky.eval()
enc_global_sun.eval()
dec_global_sky.eval()
dec_global_sun.eval()
enc_local.eval()
dec_sil.eval()
dec_app.eval()

all_dict = {}
all_dict["all"] = {}
all_dict["stat"] = {}
app_mse_list = []
app_mae_list = []
app_rmse_list = []
mse_list = []
mae_list = []
rmse_list = []
mse_global_list = []
mae_global_list = []
rmse_global_list = []
mse_mask_gt_list = []
mae_mask_gt_list = []
mse_mask_est_list = []
mae_mask_est_list = []
ldr_mse_list = []
ldr_mae_list = []
ldr_ssim_list = []
sil_mae_list = []
sil_mse_list = []

image_cnt = 0

with torch.no_grad():
    num_vis = 0
    for i, test_data in tqdm(enumerate(testLoader)):
        # retrieve the data
        global_tensor = test_data['global'].to('cuda')*4.0
        global_tensor.clamp_min_(0.0)
        global_tensor.clamp_max_(2000.0)
        local_pano = test_data['local'].to('cuda')*4.0
        local_pano.clamp_min_(0.0)
        local_pano.clamp_max_(2000.0)
        local_tensor = local_pano.clamp_max(1.0) # NOTE: for local component is OK
        local_mask = test_data['local_mask'].to('cuda')
        cosine_mask_fine = test_data['cosine_mask_fine'].to('cuda')
        sun_pos_mask_fine = test_data['sun_pos_mask_fine'].to('cuda')
        meta = test_data['metadata']

        sky_target_tensor = global_tensor.clamp_max(1.0)
        local_target_tensor = local_tensor
        mask_target_tensor = local_mask
        local_pano_target_tensor = local_pano*(1-local_mask) + local_tensor*local_mask # NOTE: important here to bound local content less than 1.0

        if args.log_image:
            global_input_tensor = linear2log(global_tensor, args.log_mu)
            sky_input_tensor = linear2log(sky_target_tensor, args.log_mu)
            local_input_tensor = recon_local_target_tensor = linear2log(local_tensor, args.log_mu)
        else:
            global_input_tensor = global_tensor
            sky_input_tensor = sky_target_tensor
            local_input_tensor = recon_local_target_tensor = local_tensor

        latent_global_sky = enc_global_sky(sky_input_tensor).detach()
        latent_global_sun = enc_global_sun(global_input_tensor).detach()
        latent_local = enc_local(local_input_tensor)
        recon_local_mask = dec_sil(latent_local)
        if args.cosine_random:
            cosine_mask_fine = torch.rand_like(cosine_mask_fine)
        if args.cosine_zero:
            cosine_mask_fine = torch.zeros_like(cosine_mask_fine)
        if args.cross_render:
            # cross render part
            swaped_latent_local = torch.clone(latent_local)
            swaped_latent_local[0::2,:] = latent_local[1::2,:]
            swaped_latent_local[1::2,:] = latent_local[0::2,:]
            latent_local = swaped_latent_local
        recon_local = dec_app(latent_local, latent_global_sky, latent_global_sun, cosine_mask_fine)
        recon_sky = dec_global_sky(latent_global_sky)
        recon_sun = dec_global_sun(latent_global_sun, sun_pos_mask_fine[:,:,:32,:])
        if args.log_image:
            image_recon_sky = log2linear(recon_sky.clamp_min(0.0).clamp_max(4.5), args.log_mu)
            image_recon_sun = log2linear(recon_sun.clamp_min(0.0).clamp_max(4.5), args.log_mu)
            image_recon_global = image_recon_sky*(1-sun_pos_mask_fine[:,:,:32,:])+ image_recon_sun
        else:
            image_recon_sky = recon_sky.clamp_min(0.0)
            image_recon_sun = recon_sun.clamp_min(0.0)
            image_recon_global = image_recon_sky*(1-sun_pos_mask_fine[:,:,:32,:])+ image_recon_sun

        if args.log_image:
            image_recon_local = log2linear(recon_local.clamp_min(0).clamp_max(4.5), args.log_mu)
        else:
            image_recon_local = recon_local.clamp_min(0)

        for k in range(image_recon_local.shape[0]):
            img_idx = image_cnt + k
            city_cam_name, sky_name, angle_id, local_id = meta[k].split()
            filebasename = '%06d_%s_%s_%s_%s'%(img_idx, city_cam_name, sky_name, angle_id, local_id)
            gt_image = local_pano_target_tensor[k:k+1] # 1, 3, 64, 128
            global_gt_image = global_tensor[k:k+1] # 1, 3, 32, 128
            global_image = image_recon_global[k:k+1] # 1, 3, 32, 128
            global_image = torch.nn.functional.pad(global_image, (0, 0, 0, 32)) # 1, 3, 64, 128
            recon_local_image = image_recon_local[k:k+1]
            mask_gt = local_mask[k:k+1]
            mask_est = (recon_local_mask[k:k+1] > 0.5).float()
            recon_image = global_image*(1-mask_est) + recon_local_image*mask_est # 1, 3, 64, 128

            gt_ldr = GammaTMO(gt_image, args.tmo_gamma, args.tmo_log_exposure) # 1, 3, 64, 128
            recon_ldr = GammaTMO(recon_image, args.tmo_gamma, args.tmo_log_exposure) # 1, 3, 64, 128

            tmp_dict = {}
            tmp_dict['APP_MAE'] = mae(gt_image*mask_gt, recon_image*mask_est).cpu().item()
            tmp_dict['APP_MSE'] = mse(gt_image*mask_gt, recon_image*mask_est).cpu().item()
            tmp_dict['APP_RMSE'] = np.sqrt(tmp_dict['APP_MSE'])
            tmp_dict['MAE'] = mae(gt_image, recon_image).cpu().item()
            tmp_dict['MSE'] = mse(gt_image, recon_image).cpu().item()
            tmp_dict['RMSE'] = np.sqrt(tmp_dict['MSE'])
            tmp_dict['MAE_GLOBAL'] = mae(global_gt_image, global_image[0,:,:32,:]).cpu().item()
            tmp_dict['MSE_GLOBAL'] = mse(global_gt_image, global_image[0,:,:32,:]).cpu().item()
            tmp_dict['RMSE_GLOBAL'] = np.sqrt(tmp_dict['MSE_GLOBAL'])
            tmp_dict['LDR_MAE'] = mae(gt_ldr, recon_ldr).cpu().item()
            tmp_dict['LDR_MSE'] = mse(gt_ldr, recon_ldr).cpu().item()
            tmp_dict['LDR_SSIM'] = ssim(gt_ldr, recon_ldr).cpu().item()
            tmp_dict['MAE_MASK_GT'] = mae(gt_image*mask_gt, recon_image*mask_gt).cpu().item()
            tmp_dict['MSE_MASK_GT'] = mse(gt_image*mask_gt, recon_image*mask_gt).cpu().item()
            tmp_dict['MAE_MASK_EST'] = mae(gt_image*mask_est, recon_image*mask_est).cpu().item()
            tmp_dict['MSE_MASK_EST'] = mse(gt_image*mask_est, recon_image*mask_est).cpu().item()
            tmp_dict['SIL_MAE'] = mae(mask_gt, mask_est).cpu().item()
            tmp_dict['SIL_MSE'] = mse(mask_gt, mask_est).cpu().item()
            all_dict["all"][filebasename] = tmp_dict
            app_mae_list.append(tmp_dict['APP_MAE'])
            app_mse_list.append(tmp_dict['APP_MSE'])
            app_rmse_list.append(tmp_dict['APP_RMSE'])
            mae_list.append(tmp_dict['MAE'])
            mse_list.append(tmp_dict['MSE'])
            rmse_list.append(tmp_dict['RMSE'])
            mae_global_list.append(tmp_dict['MAE_GLOBAL'])
            mse_global_list.append(tmp_dict['MSE_GLOBAL'])
            rmse_global_list.append(tmp_dict['RMSE_GLOBAL'])
            mae_mask_gt_list.append(tmp_dict['MAE_MASK_GT'])
            mse_mask_gt_list.append(tmp_dict['MSE_MASK_GT'])
            mae_mask_est_list.append(tmp_dict['MAE_MASK_EST'])
            mse_mask_est_list.append(tmp_dict['MSE_MASK_EST'])
            ldr_mae_list.append(tmp_dict['LDR_MAE'])
            ldr_mse_list.append(tmp_dict['LDR_MSE'])
            ldr_ssim_list.append(tmp_dict['LDR_SSIM'])
            sil_mae_list.append(tmp_dict['SIL_MAE'])
            sil_mse_list.append(tmp_dict['SIL_MSE'])

            image_gt_sample = gt_image.cpu()[0] # C, H, W
            image_recon_sample = recon_image.clamp_min(0.0).cpu()[0]
            sil_gt_sample = mask_gt.cpu()[0]
            sil_gt_sample = np.repeat(sil_gt_sample, 3, axis=0)
            sil_recon_sample = mask_est.detach().cpu()[0]
            sil_recon_sample = np.repeat(sil_recon_sample, 3, axis=0)
            cos_mask_sample = cosine_mask_fine.detach().cpu()[k]
            cos_mask_sample = np.repeat(cos_mask_sample, 3, axis=0)
            sun_posmask_sample = sun_pos_mask_fine.cpu()[k]
            sun_posmask_sample = np.repeat(sun_posmask_sample, 3, axis=0)
            image_sample = torch.cat([image_gt_sample, image_recon_sample, cos_mask_sample],  dim=2)
            image_sil_sample = torch.cat([sil_gt_sample, sil_recon_sample, sun_posmask_sample], dim=2)
            image_sample = torch.cat([image_sample, image_sil_sample], dim=1)

            ldr_image_gt_sample = gt_ldr.cpu()[0]
            ldr_image_recon_sample = recon_ldr.cpu()[0]
            ldr_image_sample = torch.cat([ldr_image_gt_sample, ldr_image_recon_sample, cos_mask_sample],  dim=2)
            ldr_image_sample = torch.cat([ldr_image_sample, image_sil_sample], dim=1)

            # NOTE: reduce disk I/O ops by merging images as subplots
            image_sample_path = os.path.join(output_image_dir, '%s_sample.hdr' %(filebasename))
            imageio.imwrite(image_sample_path, np.transpose(image_sample.cpu().numpy(), (1, 2, 0)))

            image_ldr_sample_path = os.path.join(output_image_ldr_dir, '%s_sample.png' %(filebasename))
            imageio.imwrite(image_ldr_sample_path, (np.transpose(ldr_image_sample.cpu().numpy(), (1, 2, 0))*255.0).astype(np.uint8))

        image_cnt += image_recon_local.shape[0]

avg_app_mae = np.array(app_mae_list, dtype=np.float64).mean()
avg_app_mse = np.array(app_mse_list, dtype=np.float64).mean()
avg_app_rmse = np.array(app_rmse_list, dtype=np.float64).mean()
avg_mae = np.array(mae_list, dtype=np.float64).mean()
avg_mse = np.array(mse_list, dtype=np.float64).mean()
avg_rmse = np.array(rmse_list, dtype=np.float64).mean()
avg_ldr_mae = np.array(ldr_mae_list, dtype=np.float64).mean()
avg_ldr_mse = np.array(ldr_mse_list, dtype=np.float64).mean()
avg_ldr_ssim = np.array(ldr_ssim_list, dtype=np.float64).mean()
avg_mae_mask_gt = np.array(mae_mask_gt_list, dtype=np.float64).mean()
avg_mse_mask_gt = np.array(mse_mask_gt_list, dtype=np.float64).mean()
avg_mae_mask_est = np.array(mae_mask_est_list, dtype=np.float64).mean()
avg_mse_mask_est = np.array(mse_mask_est_list, dtype=np.float64).mean()
avg_sil_mae = np.array(sil_mae_list, dtype=np.float64).mean()
avg_sil_mse = np.array(sil_mse_list, dtype=np.float64).mean()
all_dict['stat']['AVG_APP_MAE'] = avg_app_mae
all_dict['stat']['AVG_APP_MSE'] = avg_app_mse
all_dict['stat']['AVG_APP_RMSE'] = avg_app_rmse
all_dict['stat']['AVG_MAE'] = avg_mae
all_dict['stat']['AVG_MSE'] = avg_mse
all_dict['stat']['AVG_RMSE'] = avg_rmse
all_dict['stat']['AVG_LDR_MAE'] = avg_ldr_mae
all_dict['stat']['AVG_LDR_MSE'] = avg_ldr_mse
all_dict['stat']['AVG_LDR_SSIM'] = avg_ldr_ssim
all_dict['stat']['AVG_MAE_MASK_GT'] = avg_mae_mask_gt
all_dict['stat']['AVG_MSE_MASK_GT'] = avg_mse_mask_gt
all_dict['stat']['AVG_MAE_MASK_EST'] = avg_mae_mask_est
all_dict['stat']['AVG_MSE_MASK_EST'] = avg_mse_mask_est
all_dict['stat']['AVG_SIL_MAE'] = avg_sil_mae
all_dict['stat']['AVG_SIL_MSE'] = avg_sil_mse

print("avg_app_mae: %f, avg_app_mse: %f, avg_app_rmse: %f" %(avg_app_mae, avg_app_mse, avg_app_rmse))
print("avg_mae: %f, avg_mse: %f, avg_rmse: %f" %(avg_mae, avg_mse, avg_rmse))
print("avg_ldr_mae: %f, avg_ldr_mse: %f" %(avg_ldr_mae, avg_ldr_mse))
print("avg_ldr_ssim: %f" %(avg_ldr_ssim))
print("avg_mae_mask_gt: %f, avg_mse_mask_gt: %f" %(avg_mae_mask_gt, avg_mse_mask_gt))
print("avg_mae_mask_est: %f, avg_mse_mask_est: %f" %(avg_mae_mask_est, avg_mse_mask_est))
print("avg_sil_mae: %f, avg_sil_mse: %f" %(avg_sil_mae, avg_sil_mse))

with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
    f.write(json.dumps(all_dict, indent=2))

print()
print("All done. Quantitative results have been saved to %s" %(output_dir))
