import os
import argparse
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
parser.add_argument("--split_dir_syn", type=str, default='../data/synthetic/City001/split/edit')
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
parser.add_argument("--tmo_log_exposure", type=float, default=-2)
parser.add_argument("--num_loader", type=int, default=1)
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--test_split", type=str, default='test')
parser.add_argument("--model_activ", type=str, choices=['relu', 'lrelu'], default='relu')

args = parser.parse_args()

dataroot_syn = args.dataroot_syn
testsplit_syn = os.path.join(args.split_dir_syn, args.test_split+'.txt')

os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_dir), exist_ok=True)
if args.test_split != "test":
    output_dir = os.path.join(args.result_dir, args.test_split, 'edit')
else:
    output_dir = os.path.join(args.result_dir, 'edit')
os.makedirs(output_dir, exist_ok=(args.debug or args.override))
output_image_dir = os.path.join(output_dir, 'image')

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

testSampler = RandomLocalIdentiyiSampler(testSet, dataroot=dataroot_syn, batch_size=args.batch_size, num_instance=1) # a pair of same local different global for cross-render constraint

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
        _output_dir = os.path.join(output_image_dir, '%04d'%(i))
        os.makedirs(_output_dir, exist_ok=True)
        
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

        placeholder = np.zeros((64, 128, 3), dtype=np.float32)
        imageio.imwrite(os.path.join(_output_dir, '_.hdr'), placeholder)

        B, _, _, _ = global_tensor.shape

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

        recon_local_mask = dec_sil(latent_local) # B, C, H, W
        recon_local_mask[:,:,31:,:] = 1.0 # NOTE: lower semisphere always true!
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

        # save reconstructed global lighting maps
        for j in range(B):
            imageio.imwrite(os.path.join(_output_dir, '_sky_%02d.hdr' %(j)), np.transpose(image_recon_global[j].cpu().numpy(), (1, 2, 0)))

        image_recon_global = torch.nn.functional.pad(image_recon_global, (0, 0, 0, 32))

        for j in range(B):
            imageio.imwrite(os.path.join(_output_dir, 'local_%02d.hdr' %(j)), np.transpose(local_tensor[j].cpu().numpy(), (1, 2, 0)))
            repeated_latent_local = torch.repeat_interleave(latent_local[j:j+1], B, 0)
            recon_local_mask = dec_sil(repeated_latent_local)
            recon_local_mask[:,:,31:,:] = 1.0 # NOTE: lower semisphere always true!
            recon_local = dec_app(repeated_latent_local, latent_global_sky, latent_global_sun, cosine_mask_fine)
            if args.log_image:
                image_recon_local = log2linear(recon_local.clamp_min(0).clamp_max(4.5), args.log_mu)
            else:
                image_recon_local = recon_local.clamp_min(0)
            recon_image = image_recon_global*(1-recon_local_mask) + image_recon_local*recon_local_mask
            for k in range(B):
                imageio.imwrite(os.path.join(_output_dir, 'local_%02d_edit_%02d.hdr' %(j, k)), np.transpose(recon_image[k].cpu().numpy(), (1, 2, 0)))

print()
print("All done. Cross render results have been saved to %s" %(output_dir))
