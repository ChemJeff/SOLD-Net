import os
import argparse
import imageio
import torch
import time
import numpy as np
from data.dataset_synthetic_local import SynLocalDataset
from data.local_identity_sampler import RandomLocalIdentiyiSampler
from torch.utils.data import DataLoader

from model.Autoencoder import GlobalEncoder
from model.Autoencoder import LocalEncoder, LocalSilDecoder, LocalAppSplitRenderer
from utils.tonemapping import GammaTMO
from utils.loss import CosineSimilarity, NormalNLLLoss
from utils.mapping.rotation import RotateByPixel
from utils.mapping.radiometric_distorsion import RadiometricDistorsion, DistortImage
from utils.mapping.log_mapping import linear2log, log2linear
from utils.logger import *

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true")
parser.add_argument("--dataroot_syn", type=str, default="../data/synthetic")
parser.add_argument("--split_dir_syn", type=str, default='../data/synthetic/split')
parser.add_argument("--filterfile_syn", type=str, nargs='+', default=None)
parser.add_argument("--log_image", action="store_true", help="use image in log space")
parser.add_argument("--log_mu", type=float, default=16.0)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--num_epoch", type=int, required=True)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_epoch", type=int, default=0)
parser.add_argument("--load_sky_enc_path", type=str, required=True)
parser.add_argument("--load_sun_enc_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--sky_dim", type=int, default=8)
parser.add_argument("--sun_dim", type=int, default=53)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--lr_multistep_schedule", type=int, nargs='+')
parser.add_argument("--lr_multistep_gamma", type=float, default=1.0)
parser.add_argument("--l1_coeff", type=float, required=True)
parser.add_argument("--l2_coeff", type=float, required=True)
parser.add_argument("--l1_sil_coeff", type=float, required=True)
parser.add_argument("--l2_sil_coeff", type=float, required=True)
parser.add_argument("--identity_loss", action="store_true")
parser.add_argument("--identity_loss_type", type=str, default="L1", choices=['L1', 'L2', 'COS'], help='L1|L2|COS')
parser.add_argument("--identity_loss_coeff", type=float, default=0)
parser.add_argument("--cross_render_loss", action="store_true")
parser.add_argument("--cross_render_l1_coeff", type=float, default=0)
parser.add_argument("--cross_render_l2_coeff", type=float, default=0)
parser.add_argument("--tmo_gamma", type=float, default=2.2)
parser.add_argument("--tmo_log_exposure", type=float, default=-2)
parser.add_argument("--radiometric_distorsion", action="store_true")
parser.add_argument("--radiometric_distorsion_prob", type=float, default=0.5)
parser.add_argument("--rotation_distorsion", action="store_true")
parser.add_argument("--rotation_distorsion_prob", type=float, default=0.8)
parser.add_argument("--save_every", type=int, required=True)
parser.add_argument("--plot_every", type=int, required=True)
parser.add_argument("--tb_save_image_every", type=int, default=50)
parser.add_argument("--eval_every", type=int, required=True)
parser.add_argument("--num_loader", type=int, default=1)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--model_activ", type=str, choices=['relu', 'lrelu'], default='relu')

args = parser.parse_args()

dataroot_syn = args.dataroot_syn
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
enc_global_sky = GlobalEncoder(cin=3, cout=args.sky_dim, activ=args.model_activ).to('cuda')
enc_global_sky.load_state_dict(torch.load(args.load_sky_enc_path, map_location='cuda'))
enc_global_sky.eval()
enc_global_sun = GlobalEncoder(cin=3, cout=args.sun_dim, activ=args.model_activ).to('cuda')
enc_global_sun.load_state_dict(torch.load(args.load_sun_enc_path, map_location='cuda'))
enc_global_sun.eval()
enc_local = LocalEncoder(cin=3, cout=64, activ=args.model_activ).to('cuda')
dec_sil = LocalSilDecoder(cin=64, cout=1, activ=args.model_activ).to('cuda')
dec_app = LocalAppSplitRenderer(cin_l=64, cin_sky=args.sky_dim, cin_sun=args.sun_dim, cout=3, activ=args.model_activ).to('cuda')
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
BCE = torch.nn.BCELoss(reduction='mean')
CE = torch.nn.CrossEntropyLoss(reduction='mean')
COS = CosineSimilarity()
NNLL = NormalNLLLoss()
# initialize optimizer
optimizer = torch.optim.Adam([{"params": enc_local.parameters()}, 
                                {"params": dec_sil.parameters()}, 
                                {"params": dec_app.parameters()}], lr=args.lr)
lr = args.lr
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(args.lr_multistep_schedule), args.lr_multistep_gamma)

print("name: ", args.name)
print("task path: ", task_dir)

trainSet = SynLocalDataset(opt=args, dataroot=dataroot_syn, splitfile=trainsplit_syn, phase='train', filterfile=args.filterfile_syn)
valSet = SynLocalDataset(opt=args, dataroot=dataroot_syn, splitfile=valsplit_syn, phase='val', filterfile=args.filterfile_syn)

trainSampler = RandomLocalIdentiyiSampler(trainSet, dataroot=dataroot_syn, batch_size=args.batch_size, num_instance=2) # a pair of same local different global for cross-render constraint

trainLoader = DataLoader(trainSet, batch_size=args.batch_size, sampler=trainSampler, num_workers=args.num_loader, drop_last=True) # NOTE: 'shuffle' option inactive here, and 'drop_last' must be set True!
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=args.num_loader)

# by default all counter set to 0
start_epoch = 0
total_iter = 0

# if resume training
if args.resume and args.resume_epoch > 0:
    print("now resume training after epoch %d ..." %(args.resume_epoch))
    start_epoch = args.resume_epoch
    total_iter = args.resume_epoch * len(trainSet) // args.batch_size
    enc_local_load_path = os.path.join(task_dir, 'checkpoints', 'enc_local_epoch_%d' %(args.resume_epoch))
    dec_sil_load_path = os.path.join(task_dir, 'checkpoints', 'dec_sil_epoch_%d' %(args.resume_epoch))
    dec_app_load_path = os.path.join(task_dir, 'checkpoints', 'dec_app_epoch_%d' %(args.resume_epoch))
    optimizer_load_path = os.path.join(task_dir, 'checkpoints', 'optimizer_epoch_%d' %(args.resume_epoch))
    scheduler_load_path = os.path.join(task_dir, 'checkpoints', 'scheduler_epoch_%d' %(args.resume_epoch))
    print('loading local encoder from ', enc_local_load_path)
    enc_local.load_state_dict(torch.load(enc_local_load_path, map_location='cuda'))
    print('loading silhouette decoder from ', dec_sil_load_path)
    dec_sil.load_state_dict(torch.load(dec_sil_load_path, map_location='cuda'))
    print('loading appearance decoder from ', dec_app_load_path)
    dec_app.load_state_dict(torch.load(dec_app_load_path, map_location='cuda'))
    print('loading optimizer from ', scheduler_load_path)
    optimizer.load_state_dict(torch.load(optimizer_load_path, map_location='cuda'))
    print('loading scheduler from ', scheduler_load_path)
    scheduler.load_state_dict(torch.load(scheduler_load_path, map_location='cuda'))

for ep in range(start_epoch, args.num_epoch):
    epoch_start_time = time.time()
    enc_global_sky.eval()
    enc_global_sun.eval()
    enc_local.train()
    dec_sil.train()
    dec_app.train()
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
        global_tensor = train_data['global'].to('cuda')*4.0
        global_tensor.clamp_min_(0.0)
        global_tensor.clamp_max_(2000.0)
        local_tensor = train_data['local'].to('cuda')*4.0
        local_tensor.clamp_min_(0.0)
        local_tensor.clamp_max_(1.0) # NOTE: for local component is OK
        local_mask = train_data['local_mask'].to('cuda')
        cosine_mask_fine = train_data['cosine_mask_fine'].to('cuda')
        sun_pos_mask_fine = train_data['sun_pos_mask_fine'].to('cuda')

        if args.rotation_distorsion and np.random.rand() < args.rotation_distorsion_prob:
            rotate_pixels = np.random.randint(-63, 64)
            global_tensor = RotateByPixel(global_tensor, rotate_pixels)
            local_target_tensor = RotateByPixel(local_tensor, rotate_pixels)
            mask_target_tensor = RotateByPixel(local_mask, rotate_pixels)
            cosine_mask_fine = RotateByPixel(cosine_mask_fine, rotate_pixels)
            sun_pos_mask_fine = RotateByPixel(sun_pos_mask_fine, rotate_pixels)
        else:
            local_target_tensor = local_tensor
            mask_target_tensor = local_mask

        if args.radiometric_distorsion and np.random.rand() < args.radiometric_distorsion_prob:
            global_tensor, (exp_distortion, whl_distortion, gma_distortion) = RadiometricDistorsion(global_tensor)
            local_target_tensor = DistortImage(local_target_tensor, exp_distortion, whl_distortion, gma_distortion)

        if args.log_image:
            global_input_tensor = linear2log(global_tensor, args.log_mu)
            local_input_tensor = recon_local_target_tensor = linear2log(local_target_tensor, args.log_mu)
        else:
            global_input_tensor = global_tensor
            local_input_tensor = recon_local_target_tensor = local_target_tensor
        latent_global_sky = enc_global_sky(global_input_tensor.clamp_max(1.0)).detach()
        latent_global_sun = enc_global_sun(global_input_tensor).detach()
        latent_local = enc_local(local_input_tensor)
        if args.identity_loss:
            local_code_this = latent_local[0::2,:]
            local_code_other = latent_local[1::2,:]
            if args.identity_loss_type == 'L1':
                identity_loss = MAE(local_code_this, local_code_other)
            if args.identity_loss_type == 'L2':
                identity_loss = MSE(local_code_this, local_code_other)
            if args.identity_loss_type == 'COS':
                identity_loss = -COS(local_code_this, local_code_other)

        recon_local_mask = dec_sil(latent_local)
        recon_local = dec_app(latent_local, latent_global_sky, latent_global_sun, cosine_mask_fine)

        recon_mse_loss = MSE(recon_local*mask_target_tensor, recon_local_target_tensor*mask_target_tensor)
        recon_mae_loss = MAE(recon_local*mask_target_tensor, recon_local_target_tensor*mask_target_tensor)

        sil_mse_loss = MSE(recon_local_mask, mask_target_tensor)
        sil_mae_loss = MAE(recon_local_mask, mask_target_tensor)

        if args.log_image:
            image_recon_local = log2linear(recon_local.clamp_min(0), args.log_mu)
        else:
            image_recon_local = recon_local

        image_mse_loss = MSE(image_recon_local*mask_target_tensor, local_target_tensor*mask_target_tensor)
        image_mae_loss = MAE(image_recon_local*mask_target_tensor, local_target_tensor*mask_target_tensor)

        for im_idx in range(args.batch_size):
            tilmse.append(MSE(image_recon_local[im_idx]*mask_target_tensor[im_idx], local_target_tensor[im_idx]*mask_target_tensor[im_idx]).item())
            tilmae.append(MAE(image_recon_local[im_idx]*mask_target_tensor[im_idx], local_target_tensor[im_idx]*mask_target_tensor[im_idx]).item())

        if args.cross_render_loss:
            swaped_latent_local = torch.clone(latent_local)
            swaped_latent_local[0::2,:] = latent_local[1::2,:]
            swaped_latent_local[1::2,:] = latent_local[0::2,:]
            cross_recon_local = dec_app(swaped_latent_local, latent_global_sky, latent_global_sun, cosine_mask_fine)

            cross_recon_mse_loss = MSE(cross_recon_local*mask_target_tensor, recon_local_target_tensor*mask_target_tensor)
            cross_recon_mae_loss = MAE(cross_recon_local*mask_target_tensor, recon_local_target_tensor*mask_target_tensor)

        recon_loss = args.l1_coeff*recon_mae_loss + args.l2_coeff*recon_mse_loss
        recon_loss = recon_loss + args.l1_sil_coeff*sil_mae_loss + args.l2_sil_coeff*sil_mse_loss
        if args.identity_loss:
            recon_loss = recon_loss + args.identity_loss_coeff*identity_loss
        if args.cross_render_loss:
            recon_loss = recon_loss + args.cross_render_l1_coeff*cross_recon_mae_loss + args.cross_render_l2_coeff*cross_recon_mse_loss

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
                tb_logger.add_scalar("Local Loss (MSE Log)" if args.log_image else "Local Loss (MSE)", recon_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Local Loss (MAE Log)" if args.log_image else "Local Loss (MAE)", recon_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Sil Loss (MSE Log)" if args.log_image else "Sil Loss (MSE)", sil_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Sil Loss (MAE Log)" if args.log_image else "Sil Loss (MAE)", sil_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Image Loss (MSE)", image_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Image Loss (MAE)", image_mae_loss.item(), total_iter+1)
                if args.identity_loss:
                    tb_logger.add_scalar("Identity Loss (%s)" %(args.identity_loss_type), identity_loss.item(), total_iter+1)
                if args.cross_render_loss:
                    tb_logger.add_scalar("Cross Render Loss (MSE Log)" if args.log_image else "Cross Render Loss (MSE)", cross_recon_mse_loss.item(), total_iter+1)
                    tb_logger.add_scalar("Cross Render Loss (MAE Log)" if args.log_image else "Cross Render Loss (MAE)", cross_recon_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Learning Rate", lr, total_iter+1)
                tb_logger.add_scalar("Data Load Time", iter_start_time - iter_data_time, total_iter+1)
                tb_logger.add_scalar("Network Run Time", iter_net_time - iter_start_time, total_iter+1)

        if (total_iter+1)%100==0:
            ldr_gt = GammaTMO(local_target_tensor, args.tmo_gamma, args.tmo_log_exposure)
            ldr_recon = GammaTMO(image_recon_local.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)

            image_gt_sample = local_target_tensor.cpu()[0] # C, H, W
            image_recon_sample = image_recon_local.clamp_min(0.0).cpu()[0]
            sil_gt_sample = mask_target_tensor.cpu()[0]
            sil_gt_sample = np.repeat(sil_gt_sample, 3, axis=0)
            sil_recon_sample = recon_local_mask.detach().cpu()[0]
            sil_recon_sample = np.repeat(sil_recon_sample, 3, axis=0)
            cos_mask_sample = cosine_mask_fine.detach().cpu()[0]
            cos_mask_sample = np.repeat(cos_mask_sample, 3, axis=0)
            sun_posmask_sample = sun_pos_mask_fine.cpu()[0]
            sun_posmask_sample = np.repeat(sun_posmask_sample, 3, axis=0)
            image_sample = torch.cat([image_gt_sample, image_recon_sample, cos_mask_sample],  dim=2)
            image_sil_sample = torch.cat([sil_gt_sample, sil_recon_sample, sun_posmask_sample], dim=2)
            image_sample = torch.cat([image_sample, image_sil_sample], dim=1)

            ldr_image_gt_sample = ldr_gt.cpu()[0]
            ldr_image_recon_sample = ldr_recon.cpu()[0]
            ldr_image_sample = torch.cat([ldr_image_gt_sample, ldr_image_recon_sample, cos_mask_sample],  dim=2)
            ldr_image_sample = torch.cat([ldr_image_sample, image_sil_sample], dim=1)

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

    if (ep+1) % args.eval_every == 0:
        enc_local.eval()
        dec_sil.eval()
        dec_app.eval()
        print('-------------------------------------------------')
        print('            eval at %d epoch' %(ep+1))
        print('-------------------------------------------------')
        os.makedirs(os.path.join(task_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1)), exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'hdr'), exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'ldr'), exist_ok=True)
        vlmse = []
        vlmae = []
        vsilmse = []
        vsilmae = []
        vssim = []
        vilmse = []
        vilmae = []
        with torch.no_grad():
            val_start_time = time.time()
            iter_data_time = time.time()
            for i, val_data in enumerate(valLoader):
                iter_start_time = time.time()
                # retrieve the data
                global_tensor = val_data['global'].to('cuda')*4.0
                global_tensor.clamp_min_(0.0)
                global_tensor.clamp_max_(2000.0)
                local_tensor = val_data['local'].to('cuda')*4.0
                local_tensor.clamp_min_(0.0)
                local_tensor.clamp_max_(1.0)
                local_mask = val_data['local_mask'].to('cuda')
                cosine_mask_fine = val_data['cosine_mask_fine'].to('cuda')
                sun_pos_mask_fine = val_data['sun_pos_mask_fine'].to('cuda')

                local_target_tensor = local_tensor
                mask_target_tensor = local_mask

                if args.log_image:
                    global_input_tensor = linear2log(global_tensor, args.log_mu)
                    local_input_tensor = recon_local_target_tensor = linear2log(local_tensor, args.log_mu)
                else:
                    global_input_tensor = global_tensor
                    local_input_tensor = recon_local_target_tensor = local_tensor
                
                latent_global_sky = enc_global_sky(global_input_tensor.clamp_max(1.0))
                latent_global_sun = enc_global_sun(global_input_tensor)
                latent_local = enc_local(local_input_tensor)

                recon_local_mask = dec_sil(latent_local)
                recon_local = dec_app(latent_local, latent_global_sky, latent_global_sun, cosine_mask_fine)

                recon_mse_loss = MSE(recon_local*local_mask, recon_local_target_tensor*local_mask)
                recon_mae_loss = MAE(recon_local*local_mask, recon_local_target_tensor*local_mask)

                sil_mse_loss = MSE(recon_local_mask, mask_target_tensor)
                sil_mae_loss = MAE(recon_local_mask, mask_target_tensor)

                if args.log_image:
                    image_recon_local = log2linear(recon_local.clamp_min(0), args.log_mu)
                else:
                    image_recon_local = recon_local

                image_mse_loss = MSE(image_recon_local*local_mask, local_target_tensor*local_mask)
                image_mae_loss = MAE(image_recon_local*local_mask, local_target_tensor*local_mask)
                
                ldr_gt = GammaTMO(local_target_tensor, args.tmo_gamma, args.tmo_log_exposure)
                ldr_recon = GammaTMO(image_recon_local.clamp_min(0.0), args.tmo_gamma, args.tmo_log_exposure)

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
                    
                    image_gt_sample = local_target_tensor.cpu()[0] # C, H, W
                    image_recon_sample = image_recon_local.clamp_min(0.0).cpu()[0]
                    sil_gt_sample = mask_target_tensor.cpu()[0]
                    sil_gt_sample = np.repeat(sil_gt_sample, 3, axis=0)
                    sil_recon_sample = recon_local_mask.cpu()[0]
                    sil_recon_sample = np.repeat(sil_recon_sample, 3, axis=0)
                    cos_mask_sample = cosine_mask_fine.cpu()[0]
                    cos_mask_sample = np.repeat(cos_mask_sample, 3, axis=0)
                    sun_posmask_sample = sun_pos_mask_fine.cpu()[0]
                    sun_posmask_sample = np.repeat(sun_posmask_sample, 3, axis=0)
                    image_sample = torch.cat([image_gt_sample, image_recon_sample, cos_mask_sample],  dim=2)
                    image_sil_sample = torch.cat([sil_gt_sample, sil_recon_sample, sun_posmask_sample], dim=2)
                    image_sample = torch.cat([image_sample, image_sil_sample], dim=1)

                    ldr_image_gt_sample = ldr_gt.cpu()[0]
                    ldr_image_recon_sample = ldr_recon.cpu()[0]
                    ldr_image_sample = torch.cat([ldr_image_gt_sample, ldr_image_recon_sample, cos_mask_sample],  dim=2)
                    ldr_image_sample = torch.cat([ldr_image_sample, image_sil_sample], dim=1)

                    tb_logger.add_image("Val image sample %d" %(sample_idx), image_sample, ep+1)
                    tb_logger.add_image("LDR Val image sample %d" %(sample_idx), ldr_image_sample, ep+1)
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'hdr', 'sample_%d.hdr' %(sample_idx)), np.transpose(image_sample.numpy(), (1, 2, 0)), format='HDR-FI')
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'epoch_%d' %(ep+1), 'ldr', 'sample_%d.png' %(sample_idx)), (np.transpose(ldr_image_sample.numpy(), (1, 2, 0))*255.0).astype(np.uint8), format='PNG')

                vlmse.append(recon_mse_loss.item())
                vlmae.append(recon_mae_loss.item())
                vsilmse.append(sil_mse_loss.item())
                vsilmae.append(sil_mae_loss.item())
                vilmse.append(image_mse_loss.item())
                vilmae.append(image_mae_loss.item())

            vlmse_array = np.array(vlmse)
            vlmae_array = np.array(vlmae)
            vsilmse_array = np.array(vsilmse)
            vsilmae_array = np.array(vsilmae)
            vilmse_array = np.array(vilmse)
            vilmae_array = np.array(vilmae)
            vilmae_array[vilmae_array==np.inf] = np.nan
            vilmse_array[vilmse_array==np.inf] = np.nan

            if tb_logger:
                tb_logger.add_scalar("Val Local Loss (MSE Log)" if args.log_image else "Val Local Loss (MSE)", vlmse_array.mean(), ep+1)
                tb_logger.add_scalar("Val Local Loss (MAE Log)" if args.log_image else "Val Local Loss (MAE)", vlmae_array.mean(), ep+1)
                tb_logger.add_scalar("Val Sil Loss (MSE Log)" if args.log_image else "Val Sil Loss (MSE)", vsilmse_array.mean(), ep+1)
                tb_logger.add_scalar("Val Sil Loss (MAE Log)" if args.log_image else "Val Sil Loss (MAE)", vsilmae_array.mean(), ep+1)
                tb_logger.add_scalar("Val Image Loss (MAE)", np.nanmean(vilmae_array), ep+1)
                tb_logger.add_scalar("Val Image Loss (MSE)", np.nanmean(vilmse_array), ep+1)
                tb_logger.add_scalar("Val Image Loss (RMSE)", np.nanmean(np.sqrt(vilmse_array)), ep+1)

        print("Stats: MAE = %f, MSE = %f, RMSE = %f" %(np.nanmean(vilmae_array), np.nanmean(vilmse_array), np.nanmean(np.sqrt(vilmse_array))))

        print('-------------------------------------------------')
        print('            eval finish')
        print('-------------------------------------------------')

        enc_local.train()
        dec_sil.train()
        dec_app.train()

    if (ep+1) % args.save_every == 0:
        os.makedirs(os.path.join(task_dir, 'checkpoints'), exist_ok=True)
        torch.save(enc_local.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_local_latest'))
        torch.save(enc_local.state_dict(), os.path.join(task_dir, 'checkpoints', 'enc_local_epoch_%d' %(ep+1)))
        torch.save(dec_sil.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_sil_latest'))
        torch.save(dec_sil.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_sil_epoch_%d' %(ep+1)))
        torch.save(dec_app.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_app_latest'))
        torch.save(dec_app.state_dict(), os.path.join(task_dir, 'checkpoints', 'dec_app_epoch_%d' %(ep+1)))
        torch.save(optimizer.state_dict(), os.path.join(task_dir, 'checkpoints', 'optimizer_latest'))
        torch.save(optimizer.state_dict(), os.path.join(task_dir, 'checkpoints', 'optimizer_epoch_%d' %(ep+1)))
        torch.save(scheduler.state_dict(), os.path.join(task_dir, 'checkpoints', 'scheduler_latest'))
        torch.save(scheduler.state_dict(), os.path.join(task_dir, 'checkpoints', 'scheduler_epoch_%d' %(ep+1)))

    scheduler.step()
