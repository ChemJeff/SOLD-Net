import os
import argparse
import torch
import imageio
import numpy as np
from data.dataset_laval_sky import LavalSkyDataset
from data.dataset_synthetic_global import SynGlobalDataset
from data.dataset_mixed_global import MixedGlobalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.mapping.log_mapping import linear2log, log2linear
from utils.logger import *

from model.Autoencoder import GlobalEncoder, SkyDecoder, SunDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true")
parser.add_argument("--dataset", type=str, choices=['laval', 'synthetic', 'mixed'], default='mixed')
parser.add_argument("--dataroot_laval", type=str, default='../data/envmaps/LavalSkyHDR/128')
parser.add_argument("--dataroot_sun_laval", type=str, default='../data/envmaps/LavalSkyHDR/stats/128')
parser.add_argument("--dataroot_syn", type=str, default="../data/synthetic/")
parser.add_argument("--split_dir_laval", type=str, default='./data/split_laval/')
parser.add_argument("--split_dir_syn", type=str, default='../data/synthetic/split/')
parser.add_argument("--filterfile_laval", type=str, nargs='+', default=None)
parser.add_argument("--filterfile_syn", type=str, nargs='+', default=None)
parser.add_argument("--log_image", action="store_true", help="use image in log space")
parser.add_argument("--log_mu", type=float, default=16.0)
parser.add_argument("--load_sky_enc_path", type=str, required=True)
parser.add_argument("--load_sun_enc_path", type=str, required=True)
parser.add_argument("--load_sky_dec_path", type=str, required=True)
parser.add_argument("--load_sun_dec_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--sky_dim", type=int, default=16)
parser.add_argument("--sun_dim", type=int, default=45)
parser.add_argument("--num_loader", type=int, default=8)
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--test_split", type=str, default='test')
parser.add_argument("--model_activ", type=str, choices=['relu', 'lrelu'], default='relu')
parser.add_argument("--edit", type=str, choices=['sun_pos', 'sun_info', 'sky_info'])

args = parser.parse_args()
assert(args.edit in ['sun_pos', 'sun_info', 'sky_info'])

dataroot_laval = args.dataroot_laval
dataroot_syn = args.dataroot_syn
testsplit_laval = os.path.join(args.split_dir_laval, args.test_split+'.txt')
testsplit_syn = os.path.join(args.split_dir_syn, args.test_split+'.txt')

os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_dir), exist_ok=True)
if args.test_split != "test":
    output_dir = os.path.join(args.result_dir, args.test_split, 'edit_'+args.edit, args.dataset)
else:
    output_dir = os.path.join(args.result_dir, 'edit_'+args.edit, args.dataset)
os.makedirs(output_dir, exist_ok=(args.debug or args.override))
output_image_dir = os.path.join(output_dir, 'image')
os.makedirs(output_image_dir, exist_ok=(args.debug or args.override))

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
    testSet = LavalSkyDataset(opt=args, dataroot=dataroot_laval, splitfile=testsplit_laval, phase=args.test_split, filterfile=args.filterfile_laval, dataroot_sun=args.dataroot_sun_laval)
elif args.dataset == 'synthetic':
    testSet = SynGlobalDataset(opt=args, dataroot=dataroot_syn, splitfile=testsplit_syn, phase=args.test_split, filterfile=args.filterfile_syn)
elif args.dataset == 'mixed':
    testSet = MixedGlobalDataset(opt=args, dataroot_syn=dataroot_syn, splitfile_syn=testsplit_syn, dataroot_laval=dataroot_laval, splitfile_laval=testsplit_laval, dataroot_sun_laval=args.dataroot_sun_laval, phase=args.test_split, filterfile_laval=args.filterfile_laval, filterfile_syn=args.filterfile_syn)
testLoader = DataLoader(testSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_loader)

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
        _output_dir = os.path.join(output_image_dir, '%04d'%(i))
        os.makedirs(_output_dir, exist_ok=True)

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

        placeholder = np.zeros((32, 128, 3), dtype=np.float32)
        imageio.imwrite(os.path.join(_output_dir, '_.hdr'), placeholder)

        B, _, _, _ = image_tensor.shape

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
        else:
            image_recon_sky_tensor = recon_sky_tensor.clamp_min(0.0)
            image_recon_sun_tensor = recon_sun_tensor.clamp_min(0.0)
            image_recon_tensor = image_recon_sky_tensor*(1-sun_pos_mask_fine)+ image_recon_sun_tensor

        # Save editing source recon global lighting maps 
        for j in range(B):
            imageio.imwrite(os.path.join(_output_dir, '_edit_%02d.hdr' %(j)), np.transpose(image_recon_tensor[j].cpu().numpy(), (1, 2, 0)))

        # Begin editing
        for j in range(B):
            imageio.imwrite(os.path.join(_output_dir, 'global_%02d.hdr' %(j)), np.transpose(image_recon_tensor[j].cpu().numpy(), (1, 2, 0)))
            if args.edit == 'sun_pos':
                edited_sun_pos_mask_fine = sun_pos_mask_fine
                edited_latent_sun = torch.repeat_interleave(latent_sun[j:j+1], B, 0)
                edited_recon_sun_tensor = dec_sun(edited_latent_sun, edited_sun_pos_mask_fine)
                edited_image_recon_sky_tensor = torch.repeat_interleave(image_recon_sky_tensor[j:j+1], B, 0)
                if args.log_image:
                    edited_image_recon_sun_tensor = log2linear(edited_recon_sun_tensor.clamp_min(0.0).clamp_max(4.5), args.log_mu)
                    edited_image_recon_tensor = edited_image_recon_sky_tensor*(1-edited_sun_pos_mask_fine)+ edited_image_recon_sun_tensor
                else:
                    edited_image_recon_sun_tensor = edited_recon_sun_tensor.clamp_min(0.0)
                    edited_image_recon_tensor = edited_image_recon_sky_tensor*(1-edited_sun_pos_mask_fine)+ edited_image_recon_sun_tensor
            elif args.edit == 'sun_info':
                edited_sun_pos_mask_fine = torch.repeat_interleave(sun_pos_mask_fine[j:j+1], B, 0)
                edited_latent_sun = latent_sun
                edited_recon_sun_tensor = dec_sun(edited_latent_sun, edited_sun_pos_mask_fine)
                edited_image_recon_sky_tensor = torch.repeat_interleave(image_recon_sky_tensor[j:j+1], B, 0)
                if args.log_image:
                    edited_image_recon_sun_tensor = log2linear(edited_recon_sun_tensor.clamp_min(0.0).clamp_max(4.5), args.log_mu)
                    edited_image_recon_tensor = edited_image_recon_sky_tensor*(1-edited_sun_pos_mask_fine)+ edited_image_recon_sun_tensor
                else:
                    edited_image_recon_sun_tensor = edited_recon_sun_tensor.clamp_min(0.0)
                    edited_image_recon_tensor = edited_image_recon_sky_tensor*(1-edited_sun_pos_mask_fine)+ edited_image_recon_sun_tensor
            elif args.edit == 'sky_info':
                edited_sun_pos_mask_fine = torch.repeat_interleave(sun_pos_mask_fine[j:j+1], B, 0)
                edited_image_recon_sun_tensor = torch.repeat_interleave(image_recon_sun_tensor[j:j+1], B, 0)
                edited_image_recon_sky_tensor = image_recon_sky_tensor
                edited_image_recon_tensor = edited_image_recon_sky_tensor*(1-edited_sun_pos_mask_fine)+ edited_image_recon_sun_tensor
            for k in range(B):
                imageio.imwrite(os.path.join(_output_dir, 'global_%02d_edit_%s_%02d.hdr' %(j, args.edit, k)), np.transpose(edited_image_recon_tensor[k].cpu().numpy(), (1, 2, 0)))

print()
print("All done. Global editing results have been saved to %s" %(output_dir))