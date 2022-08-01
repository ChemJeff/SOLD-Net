import os
import argparse
import imageio
import torch
import time
import numpy as np
import torch.nn.functional as F
from data.dataset_laval_sky import LavalSkyDataset
from data.dataset_synthetic_global import SynGlobalDataset
from data.dataset_mixed_global import MixedGlobalDataset
from torch.utils.data import DataLoader

from model.Autoencoder import GlobalEncoder, SkyDecoder, SunDecoder
from utils.tonemapping import GammaTMO
from utils.loss import CosineSimilarity, NormalNLLLoss
from utils.mapping.radiometric_distorsion import RadiometricDistorsion, DistortImage
from utils.mapping.log_mapping import linear2log, log2linear
from utils.logger import *

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true")
parser.add_argument("--dataset", type=str, choices=['laval', 'synthetic', 'mixed'], default='laval')
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
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--num_epoch", type=int, required=True)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_epoch", type=int, default=-1)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--finetune_task_dir", type=str, default=None)
parser.add_argument("--finetune_name", type=str, default=None)
parser.add_argument("--finetune_epoch", type=int, default=-1)
parser.add_argument("--finetune_enc_sky_info_path", type=str, default=None)
parser.add_argument("--finetune_enc_sun_info_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--radiometric_distorsion", action="store_true")
parser.add_argument("--radiometric_distorsion_prob", type=float, default=0.5)
parser.add_argument("--sky_dim", type=int, default=16)
parser.add_argument("--sun_dim", type=int, default=45)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--lr_multistep_schedule", type=int, nargs='+')
parser.add_argument("--lr_multistep_gamma", type=float, default=1.0)
parser.add_argument("--l1_coeff", type=float, required=True)
parser.add_argument("--l2_coeff", type=float, required=True)
parser.add_argument("--info_loss", action="store_true")
parser.add_argument("--info_loss_type", type=str, default="NNLL", choices=['L1', 'L2', 'COS', 'CE', 'NNLL'], help='L1|L2|COS|CE|NNLL')
parser.add_argument("--info_loss_sky_coeff", type=float, default=0)
parser.add_argument("--info_loss_sun_coeff", type=float, default=0)
parser.add_argument("--tmo_gamma", type=float, default=2.2)
parser.add_argument("--tmo_log_exposure", type=float, default=-2)
parser.add_argument("--save_every", type=int, required=True)
parser.add_argument("--plot_every", type=int, required=True)
parser.add_argument("--tb_save_image_every", type=int, default=50)
parser.add_argument("--eval_every", type=int, required=True)
parser.add_argument("--num_loader", type=int, default=1)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--model_activ", type=str, choices=['relu', 'lrelu'], default='relu')

args = parser.parse_args()

dataroot_laval = args.dataroot_laval
dataroot_syn = args.dataroot_syn
trainsplit_laval = os.path.join(args.split_dir_laval, 'train.txt')
valsplit_laval = os.path.join(args.split_dir_laval, 'val.txt')
trainsplit_syn = os.path.join(args.split_dir_syn, 'train.txt')
valsplit_syn = os.path.join(args.split_dir_syn, 'val.txt')

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=(args.debug or args.override or args.resume))
task_dir = os.path.join(args.save_dir, args.name)

save_options_cmdline(task_dir, args)
logger = set_logger(task_dir)
tb_logger = set_tb_logger(task_dir)
tb_save_options_cmdline(tb_logger, args)

# initialize models
enc_sky = GlobalEncoder(cin=3, cout=args.sky_dim, activ=args.model_activ).to('cuda')
enc_sun = GlobalEncoder(cin=3, cout=args.sun_dim, activ=args.model_activ).to('cuda')
dec_sky = SkyDecoder(cin=args.sky_dim, cout=3, activ=args.model_activ).to('cuda')
dec_sun = SunDecoder(cin=args.sun_dim, cout=3, activ=args.model_activ).to('cuda')
enc_sky_info = GlobalEncoder(cin=3, cout=args.sky_dim, activ=args.model_activ).to('cuda')
enc_sun_info = GlobalEncoder(cin=3, cout=args.sun_dim, activ=args.model_activ).to('cuda')
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
BCE = torch.nn.BCELoss(reduction='mean')
CE = torch.nn.CrossEntropyLoss(reduction='mean')
COS = CosineSimilarity()
NNLL = NormalNLLLoss()
# initialize optimizer
optimizer = torch.optim.RMSprop([{"params": enc_sky.parameters()}, 
                                {"params": dec_sky.parameters()}, 
                                {"params": enc_sun.parameters()}, 
                                {"params": dec_sun.parameters()},
                                {"params": enc_sky_info.parameters(), "lr":args.lr*5},
                                {"params": enc_sun_info.parameters(), "lr":args.lr*5}], lr=args.lr, momentum=0, weight_decay=0)
lr = args.lr
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(args.lr_multistep_schedule), args.lr_multistep_gamma)

print("name: ", args.name)
print("task path: ", task_dir)

if args.dataset == 'laval':
    trainSet = LavalSkyDataset(opt=args, dataroot=dataroot_laval, splitfile=trainsplit_laval, phase='train', filterfile=args.filterfile_laval, dataroot_sun=args.dataroot_sun_laval)
    valSet = LavalSkyDataset(opt=args, dataroot=dataroot_laval, splitfile=valsplit_laval, phase='val', filterfile=args.filterfile_laval, dataroot_sun=args.dataroot_sun_laval)
elif args.dataset == 'synthetic':
    trainSet = SynGlobalDataset(opt=args, dataroot=dataroot_syn, splitfile=trainsplit_syn, phase='train', filterfile=args.filterfile_syn)
    valSet = SynGlobalDataset(opt=args, dataroot=dataroot_syn, splitfile=valsplit_syn, phase='val', filterfile=args.filterfile_syn)
elif args.dataset == 'mixed':
    trainSet = MixedGlobalDataset(opt=args, dataroot_syn=dataroot_syn, splitfile_syn=trainsplit_syn, dataroot_laval=dataroot_laval, splitfile_laval=trainsplit_laval, dataroot_sun_laval=args.dataroot_sun_laval, phase='train', filterfile_laval=args.filterfile_laval, filterfile_syn=args.filterfile_syn)
    valSet = MixedGlobalDataset(opt=args, dataroot_syn=dataroot_syn, splitfile_syn=valsplit_syn, dataroot_laval=dataroot_laval, splitfile_laval=valsplit_laval, dataroot_sun_laval=args.dataroot_sun_laval, phase='val', filterfile_laval=args.filterfile_laval, filterfile_syn=args.filterfile_syn)

trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_loader, drop_last=True) # NOTE: 'shuffle' option inactive here, and 'drop_last' must be set True!
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=args.num_loader)

# by default all counter set to 0
start_epoch = 0
total_iter = 0

# if resume training
if args.resume and args.resume_epoch > 0:
    print("now resume training after epoch %d ..." %(args.resume_epoch))
    start_epoch = args.resume_epoch
    total_iter = args.resume_epoch * len(trainSet) // args.batch_size
    enc_sky_load_path = os.path.join(task_dir, 'checkpoints', 'enc_sky_epoch_%d' %(args.resume_epoch))
    dec_sky_load_path = os.path.join(task_dir, 'checkpoints', 'dec_sky_epoch_%d' %(args.resume_epoch))
    enc_sun_load_path = os.path.join(task_dir, 'checkpoints', 'enc_sun_epoch_%d' %(args.resume_epoch))
    dec_sun_load_path = os.path.join(task_dir, 'checkpoints', 'dec_sun_epoch_%d' %(args.resume_epoch))
    enc_sky_info_load_path = os.path.join(task_dir, 'checkpoints', 'enc_sky_info_epoch_%d' %(args.resume_epoch))
    enc_sun_info_load_path = os.path.join(task_dir, 'checkpoints', 'enc_sun_info_epoch_%d' %(args.resume_epoch))
    optimizer_load_path = os.path.join(task_dir, 'checkpoints', 'optimizer_epoch_%d' %(args.resume_epoch))
    scheduler_load_path = os.path.join(task_dir, 'checkpoints', 'scheduler_epoch_%d' %(args.resume_epoch))
    print('loading sky encoder from ', enc_sky_load_path)
    enc_sky.load_state_dict(torch.load(enc_sky_load_path, map_location='cuda'))
    print('loading sky decoder from ', dec_sky_load_path)
    dec_sky.load_state_dict(torch.load(dec_sky_load_path, map_location='cuda'))
    print('loading sun encoder from ', enc_sun_load_path)
    enc_sun.load_state_dict(torch.load(enc_sun_load_path, map_location='cuda'))
    print('loading sun decoder from ', dec_sun_load_path)
    dec_sun.load_state_dict(torch.load(dec_sun_load_path, map_location='cuda'))
    print('loading sky info encoder from ', enc_sky_info_load_path)
    enc_sky_info.load_state_dict(torch.load(enc_sky_info_load_path, map_location='cuda'))
    print('loading sun info encoder from ', enc_sun_info_load_path)
    enc_sun_info.load_state_dict(torch.load(enc_sun_info_load_path, map_location='cuda'))
    print('loading optimizer from ', scheduler_load_path)
    optimizer.load_state_dict(torch.load(optimizer_load_path, map_location='cuda'))
    print('loading scheduler from ', scheduler_load_path)
    scheduler.load_state_dict(torch.load(scheduler_load_path, map_location='cuda'))

# if finetune from pretrained model
elif args.finetune:
    print("now finetuning from task %s (epoch %d) ..." %(args.finetune_name, args.finetune_epoch))
    finetune_load_dir = os.path.join(args.finetune_task_dir, args.finetune_name)
    enc_sky_load_path = os.path.join(finetune_load_dir, 'checkpoints', 'enc_sky_epoch_%d' %(args.finetune_epoch))
    dec_sky_load_path = os.path.join(finetune_load_dir, 'checkpoints', 'dec_sky_epoch_%d' %(args.finetune_epoch))
    enc_sun_load_path = os.path.join(finetune_load_dir, 'checkpoints', 'enc_sun_epoch_%d' %(args.finetune_epoch))
    dec_sun_load_path = os.path.join(finetune_load_dir, 'checkpoints', 'dec_sun_epoch_%d' %(args.finetune_epoch))
    print('loading sky encoder from ', enc_sky_load_path)
    enc_sky.load_state_dict(torch.load(enc_sky_load_path, map_location='cuda'))
    print('loading sky decoder from ', dec_sky_load_path)
    dec_sky.load_state_dict(torch.load(dec_sky_load_path, map_location='cuda'))
    print('loading sun encoder from ', enc_sun_load_path)
    enc_sun.load_state_dict(torch.load(enc_sun_load_path, map_location='cuda'))
    print('loading sun decoder from ', dec_sun_load_path)
    dec_sun.load_state_dict(torch.load(dec_sun_load_path, map_location='cuda'))
    if args.finetune_enc_sky_info_path is not None and args.finetune_enc_sun_info_path is not None:
        print('loading sky info encoder from ', args.finetune_enc_sky_info_path)
        enc_sky_info.load_state_dict(torch.load(args.finetune_enc_sky_info_path, map_location='cuda'))
        print('loading sun info encoder from ', args.finetune_enc_sun_info_path)
        enc_sun_info.load_state_dict(torch.load(args.finetune_enc_sun_info_path, map_location='cuda'))

for ep in range(start_epoch, args.num_epoch):
    epoch_start_time = time.time()
    enc_sky.train()
    dec_sky.train()
    enc_sun.train()
    dec_sun.train()
    enc_sky_info.train()
    enc_sun_info.train()
    optimizer.zero_grad()
    tilmse = []
    tilmae = []
    try:
        lr = scheduler.get_last_lr()[0]
    except Exception as e:
        lr = scheduler.get_lr()[0]
    iter_data_time = time.time()
    for i, train_data in enumerate(trainLoader):
        iter_start_time = time.time()
        # retrieve the data
        image_tensor = train_data['color'].to('cuda')
        if args.dataset == 'synthetic':
            image_tensor = image_tensor*4.0
        image_tensor.clamp_min_(0.0)
        image_tensor.clamp_max_(2000.0)
        sun_pos_mask = train_data['sun_pos_mask'].to('cuda') # B, 1, H//args.sunmask_scale, W//args.sunmask_scale
        sun_pos_mask_coarse = train_data['sun_pos_mask_coarse'].to('cuda') # B, 1, H//(2*args.sunmask_scale), W//(2*args.sunmask_scale)
        sun_pos_mask_fine = train_data['sun_pos_mask_fine'].to('cuda') # B, 1, H, W with 8 * 8 patch of ones

        if args.radiometric_distorsion and np.random.rand() < args.radiometric_distorsion_prob:
            distorted_image_tensor, (exp_distortion, whl_distortion, gma_distortion) = RadiometricDistorsion(image_tensor)
            image_target_tensor = distorted_image_tensor
            image_target_sky_tensor = DistortImage(image_tensor.clamp_max(1.0), exp_distortion, whl_distortion, gma_distortion)
        else:
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

        if args.info_loss:
            latent_sky_info = enc_sky_info(recon_sky_tensor.detach().clamp_min(0))
            latent_sun_info = enc_sun_info(recon_sun_tensor.detach().clamp_min(0))
            if args.info_loss_type == 'L1':
                info_loss_sky = MAE(latent_sky, latent_sky_info)
                info_loss_sun = MAE(latent_sun, latent_sun_info)
            if args.info_loss_type == 'L2':
                info_loss_sky = MSE(latent_sky, latent_sky_info)
                info_loss_sun = MSE(latent_sun, latent_sun_info)
            if args.info_loss_type == 'COS':
                info_loss_sky = -COS(latent_sky, latent_sky_info)
                info_loss_sun = -COS(latent_sun, latent_sun_info)
            if args.info_loss_type == 'CE':
                info_loss_sky = BCE(F.sigmoid(latent_sky_info), F.sigmoid(latent_sky.detach()))
                info_loss_sun = BCE(F.sigmoid(latent_sun_info), F.sigmoid(latent_sun.detach()))
            info_loss = args.info_loss_sky_coeff*info_loss_sky + args.info_loss_sun_coeff*info_loss_sun

        if args.log_image:
            image_recon_sky_tensor = log2linear(recon_sky_tensor.clamp_min(0.0).clamp_max(4.5), args.log_mu)
            image_recon_sun_tensor = log2linear(recon_sun_tensor.clamp_min(0.0).clamp_max(4.5), args.log_mu)
            image_recon_tensor = image_recon_sky_tensor*(1-sun_pos_mask_fine) + image_recon_sun_tensor
            recon_tensor = linear2log(image_recon_tensor, args.log_mu)
        else:
            image_recon_sky_tensor = recon_sky_tensor
            image_recon_sun_tensor = recon_sun_tensor
            recon_tensor = recon_sky_tensor*(1-sun_pos_mask_fine) + recon_sun_tensor
            image_recon_tensor = recon_tensor

        recon_mse_loss = MSE(recon_sun_tensor, recon_sun_tensor*sun_pos_mask_fine)
        recon_mae_loss = MAE(recon_sky_tensor*(1-sun_pos_mask_fine), recon_target_sky_tensor*(1-sun_pos_mask_fine)) + MAE(recon_sun_tensor*sun_pos_mask_fine, recon_target_tensor*sun_pos_mask_fine) + MAE(image_recon_tensor, image_target_tensor)

        image_mse_loss = MSE(image_recon_tensor, image_target_tensor)
        image_mae_loss = MAE(image_recon_tensor, image_target_tensor)

        for im_idx in range(args.batch_size):
            tilmse.append(MSE(image_recon_tensor[im_idx], image_target_tensor[im_idx]).item())
            tilmae.append(MAE(image_recon_tensor[im_idx], image_target_tensor[im_idx]).item())

        recon_loss = args.l1_coeff*recon_mae_loss + args.l2_coeff*recon_mse_loss
        if args.info_loss:
            recon_loss = recon_loss + info_loss

        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()

        iter_net_time = time.time()
        eta = ((iter_net_time - epoch_start_time) / (i + 1)) * len(trainLoader) - (
                    iter_net_time - epoch_start_time)

        if (total_iter+1) % args.plot_every == 0 or args.debug:
            print(
                'Name: {0} | Epoch: {1} | {2}/{3} | Iter:{4} | ReconErr: {5:.04f} | ImMSE: {6:.04f} | LR: {7:.04f} | dataT: {8:.02f} | netT: {9:.02f} | ETA: {10:02d}:{11:02d}'.format(
                    args.name, ep, i+1, len(trainLoader), total_iter+1, recon_loss.item(), image_mse_loss.item(), lr,
                                                                        iter_start_time - iter_data_time,
                                                                        iter_net_time - iter_start_time, int(eta // 60),
                    int(eta - 60 * (eta // 60))))
            if tb_logger:
                tb_logger.add_scalar("Train Loss (MSE Log)" if args.log_image else "Train Loss (MSE)", recon_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Train Loss (MAE Log)" if args.log_image else "Train Loss (MAE)", recon_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Train Loss (Log)" if args.log_image else "Train Loss", recon_loss.item(), total_iter+1)
                tb_logger.add_scalar("Image Loss (MSE)", image_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Image Loss (MAE)", image_mae_loss.item(), total_iter+1)
                if args.info_loss:
                    tb_logger.add_scalar("Info Loss (sky, %s)" %(args.info_loss_type), info_loss_sky.item(), total_iter+1)
                    tb_logger.add_scalar("Info Loss (sun, %s)" %(args.info_loss_type), info_loss_sun.item(), total_iter+1)
                tb_logger.add_scalar("Learning Rate", lr, total_iter+1)
                tb_logger.add_scalar("Data Load Time", iter_start_time - iter_data_time, total_iter+1)
                tb_logger.add_scalar("Network Run Time", iter_net_time - iter_start_time, total_iter+1)

        if (total_iter+1)%1000==0:
            ldr_gt = GammaTMO(image_target_tensor, args.tmo_gamma, args.tmo_log_exposure)
            ldr_recon = GammaTMO(image_recon_tensor.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)
            ldr_recon_sky = GammaTMO(image_recon_sky_tensor.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)
            ldr_recon_sun = GammaTMO(image_recon_sun_tensor.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)

            image_gt_sample = image_target_tensor.cpu()[0] # C, H, W
            image_recon_sample = image_recon_tensor.clamp_min(0.0).cpu()[0]
            image_recon_sky_sample = image_recon_sky_tensor.clamp_min(0.0).cpu()[0]
            image_recon_sun_sample = image_recon_sun_tensor.clamp_min(0.0).cpu()[0]
            image_sample = torch.cat([image_gt_sample, image_recon_sample],  dim=2)
            image_split_sample = torch.cat([image_recon_sky_sample, image_recon_sun_sample], dim=2)
            image_sample = torch.cat([image_sample, image_split_sample], dim=1)

            ldr_image_gt_sample = ldr_gt.cpu()[0]
            ldr_image_recon_sample = ldr_recon.cpu()[0]
            ldr_image_recon_sky_sample = ldr_recon_sky.cpu()[0]
            ldr_image_recon_sun_sample = ldr_recon_sun.cpu()[0]
            sun_pos_mask_sample = np.repeat(sun_pos_mask_fine.cpu()[0], 3, axis=0)
            ldr_image_sample = torch.cat([ldr_image_gt_sample, ldr_image_recon_sample, sun_pos_mask_sample],  dim=2)
            ldr_image_split_sample = torch.cat([ldr_image_recon_sky_sample, ldr_image_recon_sun_sample, sun_pos_mask_sample], dim=2)
            ldr_image_sample = torch.cat([ldr_image_sample, ldr_image_split_sample], dim=1)

            tb_logger.add_image("DEBUG image sample", image_sample, total_iter+1)
            tb_logger.add_image("LDR DEBUG image sample", ldr_image_sample, total_iter+1)

        total_iter += 1
        iter_data_time = time.time()

    tilmse_array = np.array(tilmse)
    tilmae_array = np.array(tilmae)
    tilmae_array[tilmae_array==np.inf] = np.nan
    tilmse_array[tilmse_array==np.inf] = np.nan

    if tb_logger:
        tb_logger.add_scalar("Train Summary (MAE)", np.nanmean(tilmae_array), ep+1)
        tb_logger.add_scalar("Train Summary (MSE)", np.nanmean(tilmse_array), ep+1)
        tb_logger.add_scalar("Train Summary (RMSE)", np.nanmean(np.sqrt(tilmse_array)), ep+1)

    print('-------------------------------------------------')
    print('           summary at %d epoch' %(ep+1))
    print('-------------------------------------------------')
    print("Stats: MAE = %f, MSE = %f, RMSE = %f" %(np.nanmean(tilmae_array), np.nanmean(tilmse_array), np.nanmean(np.sqrt(tilmse_array))))
    print('-------------------------------------------------')
    print('           summary finish')
    print('-------------------------------------------------')

    if (ep+1) % args.save_every == 0:
        os.makedirs(os.path.join(task_dir, 'checkpoints'), exist_ok=True)
        torch.save(enc_sky.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sky_latest'))
        torch.save(enc_sky.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sky_epoch_%d' %(ep+1)))
        torch.save(dec_sky.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_sky_latest'))
        torch.save(dec_sky.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_sky_epoch_%d' %(ep+1)))
        torch.save(enc_sun.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sun_latest'))
        torch.save(enc_sun.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sun_epoch_%d' %(ep+1)))
        torch.save(dec_sun.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_sun_latest'))
        torch.save(dec_sun.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_sun_epoch_%d' %(ep+1)))
        torch.save(enc_sky_info.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sky_info_latest'))
        torch.save(enc_sky_info.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sky_info_epoch_%d' %(ep+1)))
        torch.save(enc_sun_info.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sun_info_latest'))
        torch.save(enc_sun_info.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_sun_info_epoch_%d' %(ep+1)))
        torch.save(optimizer.state_dict(), os.path.join(task_dir, 'checkpoints', 'optimizer_latest'))
        torch.save(optimizer.state_dict(), os.path.join(task_dir, 'checkpoints', 'optimizer_epoch_%d' %(ep+1)))
        torch.save(scheduler.state_dict(), os.path.join(task_dir, 'checkpoints', 'scheduler_latest'))
        torch.save(scheduler.state_dict(), os.path.join(task_dir, 'checkpoints', 'scheduler_epoch_%d' %(ep+1)))

    if (ep+1) % args.eval_every == 0:
        enc_sky.eval()
        dec_sky.eval()
        enc_sun.eval()
        dec_sun.eval()
        print('-------------------------------------------------')
        print('            eval at %d epoch' %(ep+1))
        print('-------------------------------------------------')
        os.makedirs(os.path.join(task_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1)), exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'hdr'), exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'ldr'), exist_ok=True)
        vlmse = []
        vlmae = []
        vl = []
        vssim = []
        vilmse = []
        vilmae = []
        with torch.no_grad():
            val_start_time = time.time()
            iter_data_time = time.time()
            for i, val_data in enumerate(valLoader):
                iter_start_time = time.time()
                # retrieve the data
                image_tensor = val_data['color'].to('cuda')
                if args.dataset == 'synthetic':
                    image_tensor = image_tensor*4.0
                image_tensor.clamp_min_(0.0)
                image_tensor.clamp_max_(2000.0)
                sun_pos_mask = val_data['sun_pos_mask'].to('cuda') # B, 1, H//args.sunmask_scale, W//args.sunmask_scale
                sun_pos_mask_coarse = val_data['sun_pos_mask_coarse'].to('cuda') # B, 1, H//(2*args.sunmask_scale), W//(2*args.sunmask_scale)
                sun_pos_mask_fine = val_data['sun_pos_mask_fine'].to('cuda') # B, 1, H, W with 8 * 8 patch of ones

                image_target_tensor = image_tensor

                image_target_sky_tensor = image_target_tensor.clamp_max(1.0)
                image_target_sun_tensor = image_target_tensor - image_target_sky_tensor

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
                    image_recon_tensor = image_recon_sky_tensor*(1-sun_pos_mask_fine) + image_recon_sun_tensor
                    recon_tensor = linear2log(image_recon_tensor, args.log_mu)
                else:
                    image_recon_sky_tensor = recon_sky_tensor
                    image_recon_sun_tensor = recon_sun_tensor
                    recon_tensor = recon_sky_tensor*(1-sun_pos_mask_fine) + recon_sun_tensor
                    image_recon_tensor = recon_tensor

                recon_mse_loss = MSE(recon_sun_tensor, recon_sun_tensor*sun_pos_mask_fine)
                recon_mae_loss = MAE(recon_sky_tensor*(1-sun_pos_mask_fine), recon_target_sky_tensor*(1-sun_pos_mask_fine))*2 + MAE(recon_sun_tensor*sun_pos_mask_fine, recon_target_tensor*sun_pos_mask_fine) + MAE(image_recon_tensor, image_target_tensor)

                image_mse_loss = MSE(image_recon_tensor, image_tensor)
                image_mae_loss = MAE(image_recon_tensor, image_tensor)

                recon_loss = args.l1_coeff*recon_mae_loss + args.l2_coeff*recon_mse_loss
                
                ldr_gt = GammaTMO(image_tensor, args.tmo_gamma, args.tmo_log_exposure)
                ldr_recon = GammaTMO(image_recon_tensor.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)
                ldr_recon_sky = GammaTMO(image_recon_sky_tensor.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)
                ldr_recon_sun = GammaTMO(image_recon_sun_tensor.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)

                iter_net_time = time.time()
                eta = ((iter_net_time - val_start_time) / (i + 1)) * len(valLoader) - (
                            iter_net_time - val_start_time)

                print('Name: {0} | Epoch: {1} | {2}/{3} | ImMAE(Val): {4:.04f} | ImMSE(Val): {5:.04f} | dataT: {6:.02f} | netT: {7:.02f} | ETA: {8:02d}:{9:02d}'.format(
                            args.name, ep, i+1, len(valLoader), image_mae_loss.item(), image_mse_loss.item(),
                                                                                iter_start_time - iter_data_time,
                                                                                iter_net_time - iter_start_time, int(eta // 60),
                            int(eta - 60 * (eta // 60))))
                
                iter_data_time = time.time()
                
                if tb_logger and (i+1) % args.tb_save_image_every == 0:
                    sample_idx = (i+1) // args.tb_save_image_every
                    
                    image_gt_sample = image_tensor.cpu()[0] # C, H, W
                    image_recon_sample = image_recon_tensor.clamp_min(0.0).cpu()[0]
                    image_recon_sky_sample = image_recon_sky_tensor.clamp_min(0.0).cpu()[0]
                    image_recon_sun_sample = image_recon_sun_tensor.clamp_min(0.0).cpu()[0]
                    image_sample = torch.cat([image_gt_sample, image_recon_sample],  dim=2)
                    image_split_sample = torch.cat([image_recon_sky_sample, image_recon_sun_sample], dim=2)
                    image_sample = torch.cat([image_sample, image_split_sample], dim=1)

                    ldr_image_gt_sample = ldr_gt.cpu()[0]
                    ldr_image_recon_sample = ldr_recon.cpu()[0]
                    ldr_image_recon_sky_sample = ldr_recon_sky.cpu()[0]
                    ldr_image_recon_sun_sample = ldr_recon_sun.cpu()[0]
                    sun_pos_mask_sample = np.repeat(sun_pos_mask_fine.cpu()[0], 3, axis=0)
                    ldr_image_sample = torch.cat([ldr_image_gt_sample, ldr_image_recon_sample, sun_pos_mask_sample],  dim=2)
                    ldr_image_split_sample = torch.cat([ldr_image_recon_sky_sample, ldr_image_recon_sun_sample, sun_pos_mask_sample], dim=2)
                    ldr_image_sample = torch.cat([ldr_image_sample, ldr_image_split_sample], dim=1)

                    tb_logger.add_image("Val image sample %d" %(sample_idx), image_sample, ep+1)
                    tb_logger.add_image("LDR Val image sample %d" %(sample_idx), ldr_image_sample, ep+1)
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'hdr', 'sample_%d_gt.hdr' %(sample_idx)), np.transpose(image_gt_sample.numpy(), (1, 2, 0)), format='HDR-FI')
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'hdr', 'sample_%d_recon.hdr' %(sample_idx)), np.transpose(image_recon_sample.numpy(), (1, 2, 0)), format='HDR-FI')
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'ldr', 'sample_%d_gt.png' %(sample_idx)), (np.transpose(ldr_image_gt_sample.numpy(), (1, 2, 0))*255.0).astype(np.uint8), format='PNG')
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'ldr', 'sample_%d_recon.png' %(sample_idx)), (np.transpose(ldr_image_recon_sample.numpy(), (1, 2, 0))*255.0).astype(np.uint8), format='PNG')
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'ldr', 'sun_posmask_%d.png' %(sample_idx)), (np.transpose(sun_pos_mask_fine[0].cpu().numpy(), (1, 2, 0))*255.0).astype(np.uint8), format='PNG')

                vlmse.append(recon_mse_loss.item())
                vlmae.append(recon_mae_loss.item())
                vl.append(recon_loss.item())
                vilmse.append(image_mse_loss.item())
                vilmae.append(image_mae_loss.item())

            vlmse_array = np.array(vlmse)
            vlmae_array = np.array(vlmae)
            vl_array = np.array(vl)
            vilmse_array = np.array(vilmse)
            vilmae_array = np.array(vilmae)
            vilmae_array[vilmae_array==np.inf] = np.nan
            vilmse_array[vilmse_array==np.inf] = np.nan

            if tb_logger:
                tb_logger.add_scalar("Val Loss (MSE Log)" if args.log_image else "Val Loss (MSE)", vlmse_array.mean(), ep+1)
                tb_logger.add_scalar("Val Loss (MAE Log)" if args.log_image else "Val Loss (MAE)", vlmae_array.mean(), ep+1)
                tb_logger.add_scalar("Val Loss (Log)" if args.log_image else "Val Loss", vl_array.mean(), ep+1)
                tb_logger.add_scalar("Val Image Loss (MAE)", np.nanmean(vilmae_array), ep+1)
                tb_logger.add_scalar("Val Image Loss (MSE)", np.nanmean(vilmse_array), ep+1)
                tb_logger.add_scalar("Val Image Loss (RMSE)", np.nanmean(np.sqrt(vilmse_array)), ep+1)

        print("Stats: MAE = %f, MSE = %f, RMSE = %f" %(np.nanmean(vilmae_array), np.nanmean(vilmse_array), np.nanmean(np.sqrt(vilmse_array))))

        print('-------------------------------------------------')
        print('            eval finish')
        print('-------------------------------------------------')

    scheduler.step()
