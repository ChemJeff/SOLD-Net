import os
import pickle
import cv2
import argparse
import time
import imageio
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from data.dataset_synthetic import SyntheticDataset
from torch.utils.data import DataLoader

from model.Estimator import SplitLightEstimator
from utils.tonemapping import GammaTMO
from utils.loss import mae, mse
from utils.loss.pytorch_ssim import ssim
from utils.loss import CosineSimilarity, NormalNLLLoss
from utils.mapping.radiometric_distorsion import GetDistortConfig, DistortImage
from utils.metrics import calc_azimuth_error
from utils.mapping.log_mapping import linear2log, log2linear
from utils.logger import *

colors = [(0, 0, 255), # RED
          (255, 255, 0), # CYAN
          (255, 0, 255), # PURPLE
          (0, 255, 255)] # YELLOW

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true")

parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument("--gpu_list", type=int, nargs='+')

parser.add_argument("--dataroot_syn", type=str, default="../data/synthetic")
parser.add_argument("--split_dir_syn", type=str, default='../data/synthetic/split')
parser.add_argument("--filterfile_syn", type=str, nargs='+', default=None)

parser.add_argument("--load_app_dec_path", type=str, required=True)
parser.add_argument("--load_sil_dec_path", type=str, required=True)
parser.add_argument("--load_sky_dec_path", type=str, required=True)
parser.add_argument("--load_sun_dec_path", type=str, required=True)

parser.add_argument("--log_image", action="store_true", help="use image in log space")
parser.add_argument("--log_mu", type=float, default=16.0)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--num_epoch", type=int, required=True)

parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_epoch", type=int, default=0)
parser.add_argument("--resume_iter", type=int, default=0)

parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)

parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--lr_multistep_schedule", type=int, nargs='+')
parser.add_argument("--lr_multistep_gamma", type=float, default=1.0)

parser.add_argument("--mask_mse_coeff", type=float, required=True)
parser.add_argument("--pos_bce_coeff", type=float, required=True)
parser.add_argument("--sky_code_mse_coeff", type=float, required=True)
parser.add_argument("--sun_code_mse_coeff", type=float, required=True)
parser.add_argument("--local_code_mse_coeff", type=float, required=True)
parser.add_argument("--sky_mae_coeff", type=float, required=True)
parser.add_argument("--sun_mae_coeff", type=float, required=True)
parser.add_argument("--local_app_mae_coeff", type=float, required=True)
parser.add_argument("--local_sil_mse_coeff", type=float, required=True)

parser.add_argument("--ldr_distorsion", action='store_true')
parser.add_argument("--ldr_distorsion_prob", type=float, default=1.0)
parser.add_argument("--tmo_gamma", type=float, default=2.2)
parser.add_argument("--tmo_log_exposure", type=float, default=-2)
parser.add_argument("--save_every", type=int, required=True)
parser.add_argument("--save_every_iter", type=int, default=None)
parser.add_argument("--plot_every_iter", type=int, required=True)
parser.add_argument("--tb_save_image_every", type=int, default=100)
parser.add_argument("--eval_every", type=int, required=True)
parser.add_argument("--eval_every_iter", type=int, default=None)
parser.add_argument("--num_loader", type=int, default=1)
parser.add_argument("--override_scheduler", action="store_true")
parser.add_argument("--override_optimizer", action="store_true")

def main():
    args = parser.parse_args()

    assert(not (args.resume_epoch and args.resume_iter))

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
    estimator = SplitLightEstimator().to('cuda')
    print('loading global sky decoder from ', args.load_sky_dec_path)
    estimator.sky_decoder.load_state_dict(torch.load(args.load_sky_dec_path, map_location='cuda'))
    estimator.sky_decoder.eval()
    print('loading global sun decoder from ', args.load_sun_dec_path)
    estimator.sun_decoder.load_state_dict(torch.load(args.load_sun_dec_path, map_location='cuda'))
    estimator.sun_decoder.eval()
    print('loading local app renderer from ', args.load_app_dec_path)
    estimator.local_app_render.load_state_dict(torch.load(args.load_app_dec_path, map_location='cuda'))
    estimator.local_app_render.eval()
    print('loading local sil decoder from ', args.load_sil_dec_path)
    estimator.local_sil_decoder.load_state_dict(torch.load(args.load_sil_dec_path, map_location='cuda'))
    estimator.local_sil_decoder.eval()
    MSE = torch.nn.MSELoss(reduction='mean')
    MAE = torch.nn.L1Loss(reduction='mean')
    BCE = torch.nn.BCELoss(reduction='mean')
    CE = torch.nn.CrossEntropyLoss(reduction='mean')
    COS = CosineSimilarity()
    NNLL = NormalNLLLoss()

    # initialize optimizer
    optimizer = torch.optim.Adam([{"params": estimator.feat_exactor.parameters()}, 
                                    {"params": estimator.mask_decoder.parameters()},
                                    {"params": estimator.pos_decoder.parameters()},
                                    {"params": estimator.sky_estimator.parameters()},
                                    {"params": estimator.sun_estimator.parameters()},
                                    {"params": estimator.local_extractor.parameters()},
                                    {"params": estimator.local_estimator.parameters()}], lr=args.lr)
    lr = args.lr
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(args.lr_multistep_schedule), args.lr_multistep_gamma)

    print("name: ", args.name)
    print("task path: ", task_dir)

    trainSet = SyntheticDataset(opt=args, dataroot=dataroot_syn, splitfile=trainsplit_syn, phase='train', filterfile=args.filterfile_syn)
    valSet = SyntheticDataset(opt=args, dataroot=dataroot_syn, splitfile=valsplit_syn, phase='val', filterfile=args.filterfile_syn)

    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_loader, drop_last=True) # NOTE: 'shuffle' option inactive here, and 'drop_last' must be set True!
    valLoader = DataLoader(valSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_loader, drop_last=False)

    # by default all counter set to 0
    start_epoch = 0
    total_iter = 0

    # if resume training
    if args.resume and args.resume_epoch > 0:
        print("now resume training after epoch %d ..." %(args.resume_epoch))
        start_epoch = args.resume_epoch
        total_iter = args.resume_epoch * len(trainSet) // args.batch_size
        estimator_load_path = os.path.join(task_dir, 'checkpoints', 'estimator_epoch_%d' %(args.resume_epoch))
        optimizer_load_path = os.path.join(task_dir, 'checkpoints', 'optimizer_epoch_%d' %(args.resume_epoch))
        scheduler_load_path = os.path.join(task_dir, 'checkpoints', 'scheduler_epoch_%d' %(args.resume_epoch))
        print('loading estimator from ', estimator_load_path)
        estimator.load_state_dict(torch.load(estimator_load_path, map_location='cuda'))
        if not args.override_optimizer:
            print('loading optimizer from ', scheduler_load_path)
            optimizer.load_state_dict(torch.load(optimizer_load_path, map_location='cuda'))
        if not args.override_scheduler:
            print('loading scheduler from ', scheduler_load_path)
            scheduler.load_state_dict(torch.load(scheduler_load_path, map_location='cuda'))
        else:
            for pre_ep in range(start_epoch):
                optimizer.zero_grad()
                optimizer.step()
                scheduler.step()

    if args.resume and args.resume_iter > 0:
        print("now resume training after iter %d ..." %(args.resume_iter))
        start_epoch = args.resume_iter // (len(trainSet) // args.batch_size)
        total_iter = args.resume_iter
        estimator_load_path = os.path.join(task_dir, 'checkpoints', 'estimator_iter_%d' %(args.resume_iter))
        optimizer_load_path = os.path.join(task_dir, 'checkpoints', 'optimizer_iter_%d' %(args.resume_iter))
        scheduler_load_path = os.path.join(task_dir, 'checkpoints', 'scheduler_iter_%d' %(args.resume_iter))
        print('loading estimator from ', estimator_load_path)
        estimator.load_state_dict(torch.load(estimator_load_path, map_location='cuda'))
        if not args.override_optimizer:
            print('loading optimizer from ', scheduler_load_path)
            optimizer.load_state_dict(torch.load(optimizer_load_path, map_location='cuda'))
        if not args.override_scheduler:
            print('loading scheduler from ', scheduler_load_path)
            scheduler.load_state_dict(torch.load(scheduler_load_path, map_location='cuda'))
        else:
            for pre_ep in range(start_epoch):
                optimizer.zero_grad()
                optimizer.step()
                scheduler.step()

    # load single card version first then dataparallel 
    if args.multi_gpu:
        device_ids = list(args.gpu_list)
        print("[INFO] now using %d GPU(s): " %(len(device_ids)), device_ids)
        estimator = torch.nn.DataParallel(estimator, device_ids=device_ids)
        # optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
        num_gpus = len(device_ids)
    else:
        num_gpus = 1

    for ep in range(start_epoch, args.num_epoch):
        epoch_start_time = time.time()

        if args.multi_gpu:
            estimator.module.feat_exactor.train()
            estimator.module.mask_decoder.train()
            estimator.module.pos_decoder.train()
            estimator.module.sky_estimator.train()
            estimator.module.sun_estimator.train()
            estimator.module.local_extractor.train()
            estimator.module.local_estimator.train()
        else:
            estimator.feat_exactor.train()
            estimator.mask_decoder.train()
            estimator.pos_decoder.train()
            estimator.sky_estimator.train()
            estimator.sun_estimator.train()
            estimator.local_extractor.train()
            estimator.local_estimator.train()
        optimizer.zero_grad()

        gmse = []
        gmae = []
        gssim = []
        lmae = []
        lmse = []
        lssim = []
        az = []
        el = []
        try:
            lr = scheduler.get_last_lr()[0]
        except Exception as e:
            lr = scheduler.get_lr()[0]

        iter_data_time = time.time()
        for i, train_data in enumerate(trainLoader):
            iter_start_time = time.time()
            # retrieve the data
            persp_tensor = train_data['color'].to('cuda') # B, 3, 240, 320
            global_tensor = train_data['global_lighting'].to('cuda') # B, 3, 32, 128
            global_tensor.clamp_min_(0.0)
            global_tensor.clamp_max_(2000.0)
            global_sky_code = train_data['global_sky_code'].to('cuda') # B, 16
            global_sun_code = train_data['global_sun_code'].to('cuda') # B, 45
            local_pano = train_data['local_pano'].to('cuda') # B, 3, 64, 128
            local_pano.clamp_min_(0.0)
            local_pano.clamp_max_(2000.0)
            local_tensor = local_pano.clamp_max(1.0) # NOTE: for local component is OK
            local_mask = train_data['local_mask'].to('cuda') # B, 4, 1, 64, 128
            local_pos = train_data['local_pos'].to('cuda') # B, 4, 2
            local_code = train_data['local_code'].to('cuda') # B, 4, 64
            cosine_mask_fine = train_data['cosine_mask_fine'].to('cuda') # B, 1, 64, 128
            sun_vis = train_data['is_sunny'].to('cuda') # B
            sun_azimuth = train_data['sun_azimuth'] # B
            sun_elevation = train_data['sun_elevation'] # B
            sun_pos_mask = train_data['sun_pos_mask'].to('cuda') # B, 1, 8, 32
            sun_pos_mask_fine = train_data['sun_pos_mask_fine'].to('cuda') # B, 1, 32, 128
            persp_shadow_mask = train_data['persp_shadow_mask'].to('cuda') # B, 1, 240, 320

            local_pano_target_tensor = local_pano*(1-local_mask) + local_tensor*local_mask # NOTE: important here to bound local content less than 1.0

            # NOTE: ldr intensity distorsion to more generalize to real data ldr pictures
            # accurate sun position is more useful than local accuracy!
            if args.ldr_distorsion and np.random.rand() < args.ldr_distorsion_prob:
                exp_distortion, whl_distortion, gma_distortion = GetDistortConfig(persp_tensor, exp_mean=0.1, exp_std=0.2, exp_lower=0.6, exp_upper=2.5)
                distorted_persp_tensor = DistortImage(persp_tensor, exp_distortion, whl_distortion, gma_distortion)
                mask_est, pos_est = estimator(distorted_persp_tensor, global_only=True)
                mask_mse_loss = MSE(mask_est, persp_shadow_mask)
                pos_bce_loss_split = F.binary_cross_entropy(pos_est, sun_pos_mask, reduction="none").sum(dim=[1, 2, 3])
                
                pos_bce_loss = (pos_bce_loss_split*sun_vis).mean()

                global_loss = \
                    args.mask_mse_coeff* mask_mse_loss + \
                    args.pos_bce_coeff* pos_bce_loss

                optimizer.zero_grad()
                global_loss.backward()
                optimizer.step()

            mask_est, pos_est, sky_code_est, sun_code_est, local_code_est, sky_est_raw, sun_est_raw, local_app_est_raw, local_sil_est, azimuth_deg_est, elevation_deg_est, pos_est_fine, cosine_mask_fine_est = estimator(persp_tensor, local_pos, global_sky_code, global_sun_code, cosine_mask_fine)

            B, num_local, D = local_pos.shape # B, num_local, 2

            azimuth_deg_gt = sun_azimuth / np.pi * 180
            elevation_deg_gt = sun_elevation / np.pi * 180
            azimuth_deg_est = azimuth_deg_est.cpu()
            elevation_deg_est = elevation_deg_est.cpu()

            # decode codes to images
            sky_est = log2linear(sky_est_raw.clamp_min(0.0).clamp_max(4.5), args.log_mu) # B, 3, 32, 128
            sun_est = log2linear(sun_est_raw.clamp_min(0.0).clamp_max(4.5), args.log_mu) # B, 3, 32, 128
            
            local_app_est = log2linear(local_app_est_raw.clamp_min(0.0), args.log_mu) # B*num_local, 3, 64, 128
            local_app_est = local_app_est.view(B, num_local, 3, 64 ,128) # B, num_local, 3, 64, 128
            local_sil_est = local_sil_est.view(B, num_local, 1, 64 ,128) # B, num_local, 1, 64, 128

            # combine to the full local lighting map
            global_est = sky_est*(1-pos_est_fine) + sun_est # B, 3, 32, 128
            local_est = torch.nn.functional.pad(global_est, (0, 0, 0, 32)).unsqueeze(1)*(1-local_sil_est) + local_app_est*local_sil_est # B, 4, 3, 64, 128
            
            # calc losses
            # mask_bce_loss = BCE(mask_est.view(args.batch_size, -1), persp_shadow_mask.view(args.batch_size, -1))
            mask_mse_loss = MSE(mask_est, persp_shadow_mask)
            pos_bce_loss_split = F.binary_cross_entropy(pos_est, sun_pos_mask, reduction="none").sum(dim=[1, 2, 3])
            sky_code_mse_loss = MSE(sky_code_est, global_sky_code)
            sun_code_mse_loss = MSE(sun_code_est, global_sun_code)
            local_code_mse_loss = MSE(local_code_est, local_code.view(-1, 64))
            sky_mae_loss = MAE(sky_est, global_tensor.clamp_max(1.0))
            sun_mae_loss = MAE(sun_est, global_tensor*sun_pos_mask_fine)
            local_app_mae_loss = MAE(local_app_est*local_mask, local_tensor*local_mask)
            local_sil_mse_loss = MSE(local_sil_est, local_mask)

            pos_bce_loss = (pos_bce_loss_split*sun_vis).mean()

            # combine losses
            loss = \
                    args.mask_mse_coeff* mask_mse_loss + \
                    args.pos_bce_coeff* pos_bce_loss + \
                    args.sky_code_mse_coeff* sky_code_mse_loss + \
                    args.sun_code_mse_coeff* sun_code_mse_loss + \
                    args.local_code_mse_coeff* local_code_mse_loss + \
                    args.sky_mae_coeff* sky_mae_loss + \
                    args.sun_mae_coeff* sun_mae_loss + \
                    args.local_app_mae_coeff* local_app_mae_loss + \
                    args.local_sil_mse_coeff* local_sil_mse_loss

            ldr_global = GammaTMO(global_tensor.cpu(), args.tmo_gamma, args.tmo_log_exposure)
            ldr_local = GammaTMO(local_pano_target_tensor.cpu(), args.tmo_gamma, args.tmo_log_exposure)

            ldr_global_est = GammaTMO(global_est.detach().cpu(), args.tmo_gamma, args.tmo_log_exposure)
            ldr_local_est = GammaTMO(local_est.detach().cpu(), args.tmo_gamma, args.tmo_log_exposure)

            global_image_mae_loss = MAE(global_est, global_tensor)
            global_image_mse_loss = MSE(global_est, global_tensor)
            global_image_ssim_loss = ssim(ldr_global_est, ldr_global)
            local_image_mae_loss = MAE(local_est, local_pano_target_tensor)
            local_image_mse_loss = MSE(local_est, local_pano_target_tensor)
            local_image_ssim_loss = ssim(ldr_local_est.view(B*num_local, 3, 64 ,128), ldr_local.view(B*num_local, 3, 64 ,128))

            # train loss stats
            for im_idx in range(args.batch_size):
                gmse.append(MSE(global_est[im_idx], global_tensor[im_idx]).item())
                gmae.append(MAE(global_est[im_idx], global_tensor[im_idx]).item())
                gssim.append(ssim(ldr_global_est[im_idx:im_idx+1], ldr_global[im_idx:im_idx+1]).item())
                for local_idx in range(num_local):
                    lmse.append(MSE(local_est[im_idx][local_idx], local_pano_target_tensor[im_idx][local_idx]).item())
                    lmae.append(MAE(local_est[im_idx][local_idx], local_pano_target_tensor[im_idx][local_idx]).item())
                    lssim.append(ssim(ldr_local_est[im_idx][local_idx:local_idx+1], ldr_local[im_idx][local_idx:local_idx+1]).item())
                if sun_vis[im_idx] == 1:
                    az.append(calc_azimuth_error(azimuth_deg_est[im_idx], azimuth_deg_gt[im_idx], unit='deg'))
                    el.append(elevation_deg_est[im_idx] - elevation_deg_gt[im_idx])

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (i + 1)) * len(trainLoader) - (
                        iter_net_time - epoch_start_time)

            if (total_iter+1) % args.plot_every_iter == 0 or args.debug:
                print(
                    'Name:{0}|Epoch:{1}|{2}/{3}|Iter:{4}|m:{5:.04f}|p:{6:.04f}|kc:{7:.04f}|sc:{8:.04f}|lc:{9:.04f}|k:{10:.04f}|s:{11:.04f}|la:{12:04f}|ls:{13:.04f}|LR:{14:.04f}|dataT:{15:.02f}|netT:{16:.02f}|ETA:{17:02d}:{18:02d}'.format(
                        args.name, ep, i+1, len(trainLoader), total_iter+1, mask_mse_loss.item(), pos_bce_loss.item(),
                        sky_code_mse_loss.item(), sun_code_mse_loss.item(), local_code_mse_loss.item(),
                        sky_mae_loss.item(), sun_mae_loss.item(), local_app_mae_loss.item(), local_sil_mse_loss.item(),
                        lr, iter_start_time - iter_data_time, iter_net_time - iter_start_time,
                        int(eta // 60), int(eta - 60 * (eta // 60))
                    )
                )

            if tb_logger:
                tb_logger.add_scalar("Shadow Loss (MSE)", mask_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Pos Loss (BCE)", pos_bce_loss.item(), total_iter+1)
                tb_logger.add_scalar("Sky Code Loss (MSE)", sky_code_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Sun Code Loss (MSE)", sun_code_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Local Code Loss (MSE)", local_code_mse_loss, total_iter+1)
                tb_logger.add_scalar("Sky Loss (MAE)", sky_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Sun Loss (MAE)", sun_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Local App Loss (MAE)", local_app_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Local Sil Loss (MSE)", local_sil_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Global Image Loss (MSE)", global_image_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Global Image Loss (MAE)", global_image_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Global Image Loss (SSIM)", global_image_ssim_loss.item(), total_iter+1)
                tb_logger.add_scalar("Local Image Loss (MSE)", local_image_mse_loss.item(), total_iter+1)
                tb_logger.add_scalar("Local Image Loss (MAE)", local_image_mae_loss.item(), total_iter+1)
                tb_logger.add_scalar("Local Image Loss (SSIM)", local_image_ssim_loss.item(), total_iter+1)
                tb_logger.add_scalar("Learning Rate", lr, total_iter+1)
                tb_logger.add_scalar("Data Load Time", iter_start_time - iter_data_time, total_iter+1)
                tb_logger.add_scalar("Network Run Time", iter_net_time - iter_start_time, total_iter+1)

            if (total_iter+1)%500 == 0:
                persp_sample = np.transpose(persp_tensor.cpu().numpy()[0], (1, 2, 0)) # 240, 320, 3
                persp_shadow_sample = np.transpose(persp_shadow_mask.cpu().numpy()[0], (1, 2, 0)) # 240, 320, 1
                persp_shadow_sample = np.repeat(persp_shadow_sample, 3, axis=2) # 240, 320, 3
                local_pos_sample = local_pos[0].cpu().numpy()
                persp_loc_vis_sample = cv2.cvtColor((persp_sample*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                for local_idx in range(4):
                    if local_pos_sample[local_idx][0] < 0 or local_pos_sample[local_idx][1] < 0:
                        local_pos_sample[local_idx] = -local_pos_sample[local_idx] - 1
                    cv2.circle(persp_loc_vis_sample, (int(local_pos_sample[local_idx][1]), int(local_pos_sample[local_idx][0])), 1, colors[local_idx], 4)
                persp_loc_vis_sample = cv2.cvtColor(persp_loc_vis_sample, cv2.COLOR_BGR2RGB)/255.0 # 240, 320, 3
                cos_mask_sample = cosine_mask_fine.cpu().numpy()[0]
                cos_mask_sample = np.repeat(cos_mask_sample, 3, axis=0)
                cos_mask_sample = np.transpose(cos_mask_sample, (1, 2, 0)) # 64, 128, 3
                sun_posmask_sample = sun_pos_mask_fine.cpu().numpy()[0] # 1, 32, 128
                sun_posmask_sample = np.repeat(sun_posmask_sample, 3, axis=0) # 3, 32, 128
                sun_posmask_sample = np.transpose(sun_posmask_sample, (1, 2, 0)) # 32, 128, 3
                global_sample = np.transpose(ldr_global.cpu().numpy()[0], (1, 2, 0)) # 32, 128, 3
                local_sil_sample = np.transpose(local_mask.cpu().numpy()[0], (0, 2, 3, 1)) # 4, 64, 128, 1
                local_sil_sample = np.repeat(local_sil_sample, 3, axis=3) # 4, 64, 128, 3
                local_sample = np.transpose(ldr_local.cpu().numpy()[0], (0, 2, 3, 1)) # 4, 64, 128, 3
                local_sample_stitched = np.concatenate([
                    np.concatenate([local_sample[1], local_sample[0], local_sil_sample[1], local_sil_sample[0]], axis=1),
                    np.concatenate([local_sample[2], local_sample[3], local_sil_sample[2], local_sil_sample[3]], axis=1)
                ], axis=0)
                global_sample_stitched = np.concatenate([
                    np.concatenate([global_sample, sun_posmask_sample], axis=0),
                    cos_mask_sample
                ], axis=0)
                # persp_sample_stitched = np.concatenate([persp_sample, persp_shadow_sample, persp_loc_vis_sample], axis=1)

                global_est_sample = np.transpose(ldr_global_est.detach().cpu().numpy()[0], (1, 2, 0)) # 32, 128, 3
                sun_posmask_est_sample = np.transpose(pos_est_fine.detach().cpu().numpy()[0], (1, 2, 0)) # 32, 128, 1
                sun_posmask_est_sample = np.repeat(sun_posmask_est_sample, 3, axis=2)
                cos_mask_est_sample = cosine_mask_fine_est.detach().cpu().numpy()[0] # 1, 64, 128
                cos_mask_est_sample = np.repeat(cos_mask_est_sample, 3, axis=0)
                cos_mask_est_sample = np.transpose(cos_mask_est_sample, (1, 2, 0)) # 64, 128, 3

                local_sil_est_sample = np.transpose(local_sil_est.detach().cpu().numpy()[0], (0, 2, 3, 1)) # 4, 64, 128, 1
                local_sil_est_sample = np.repeat(local_sil_est_sample, 3, axis=3) # 4, 64, 128, 3
                local_est_sample = np.transpose(ldr_local_est.cpu().numpy()[0], (0, 2, 3, 1)) # 4, 64, 128, 3

                persp_shadow_est_sample = np.transpose(mask_est.detach().cpu().numpy()[0], (1, 2, 0)) # 240, 320, 1
                persp_shadow_est_sample = np.repeat(persp_shadow_est_sample, 3, axis=2) # 240, 320, 3

                local_est_sample_stitched = np.concatenate([
                    np.concatenate([local_est_sample[1], local_est_sample[0], local_sil_est_sample[1], local_sil_est_sample[0]], axis=1),
                    np.concatenate([local_est_sample[2], local_est_sample[3], local_sil_est_sample[2], local_sil_est_sample[3]], axis=1)
                ], axis=0)
                global_est_sample_stitched = np.concatenate([
                    np.concatenate([global_est_sample, sun_posmask_est_sample], axis=0),
                    cos_mask_est_sample
                ], axis=0)

                pano_sample_stitched = np.concatenate([
                    np.concatenate([global_sample_stitched, local_sample_stitched], axis=1),
                    np.concatenate([global_est_sample_stitched, local_est_sample_stitched], axis=1)
                ], axis=0)
                persp_sample_stitched = np.concatenate([persp_loc_vis_sample, persp_shadow_sample, persp_shadow_est_sample], axis=1)

                tb_logger.add_image("LDR DEBUG pano sample", np.transpose(pano_sample_stitched, [2, 0 ,1]), total_iter+1)
                tb_logger.add_image("LDR DEBUG perspective sample", np.transpose(persp_sample_stitched, [2, 0, 1]), total_iter+1)

            if (total_iter+1) % args.save_every_iter == 0:
                save_model(args, task_dir, estimator, optimizer, scheduler, iter=total_iter+1)

            if (total_iter+1) % args.eval_every_iter == 0:
                validate_model(args, task_dir, tb_logger, valLoader, estimator, total_iter+1)

            total_iter += 1
            iter_data_time = time.time()

        gmse_array = np.array(gmse)
        gmae_array = np.array(gmae)
        gssim_array = np.array(gssim)
        lmse_array = np.array(lmse)
        lmae_array = np.array(lmae)
        lssim_array = np.array(lssim)
        az_array = np.array(az)
        el_array = np.array(el)

        if tb_logger:
            tb_logger.add_scalar("Train Summary (G-MAE)", np.nanmean(gmae_array), ep+1)
            tb_logger.add_scalar("Train Summary (G-MSE)", np.nanmean(gmse_array), ep+1)
            tb_logger.add_scalar("Train Summary (G-RMSE)", np.nanmean(np.sqrt(gmse_array)), ep+1)
            tb_logger.add_scalar("Train Summary (G-SSIM)", np.nanmean(gssim_array), ep+1)
            tb_logger.add_scalar("Train Summary (L-MAE)", np.nanmean(lmae_array), ep+1)
            tb_logger.add_scalar("Train Summary (L-MSE)", np.nanmean(lmse_array), ep+1)
            tb_logger.add_scalar("Train Summary (L-RMSE)", np.nanmean(np.sqrt(lmse_array)), ep+1)
            tb_logger.add_scalar("Train Summary (L-SSIM)", np.nanmean(lssim_array), ep+1)
            tb_logger.add_scalar("Train Summary (AZ-d)", np.nanmean(abs(az_array)), ep+1)
            tb_logger.add_scalar("Train Summary (EL-d)", np.nanmean(abs(el_array)), ep+1)
            add_tf_summary_histogram(tb_logger, "Train azimuth_error(deg)", az_array, ep+1)
            add_tf_summary_histogram(tb_logger, "Train elevation_error(deg)", el_array, ep+1)

        print('-------------------------------------------------')
        print('           summary at %d epoch' %(ep+1))
        print('-------------------------------------------------')
        print("Stats: G-MAE = %f, G-MSE = %f, G-RMSE = %f, G-SSIM = %f" %(np.nanmean(gmae_array), np.nanmean(gmse_array), np.nanmean(np.sqrt(gmse_array)), np.nanmean(gssim_array)))
        print("Stats: L-MAE = %f, L-MSE = %f, L-RMSE = %f, L-SSIM = %f" %(np.nanmean(lmae_array), np.nanmean(lmse_array), np.nanmean(np.sqrt(lmse_array)), np.nanmean(lssim_array)))
        print("Stats: AZ-d = %f, EL-d = %f" %(np.nanmean(abs(az_array)), np.nanmean(abs(el_array))))
        print('-------------------------------------------------')
        print('           summary finish')
        print('-------------------------------------------------')    

        scheduler.step() # NOTE: important!

        if (ep+1) % args.save_every == 0:
            save_model(args, task_dir, estimator, optimizer, scheduler, epoch=ep+1)

        if (ep+1) % args.eval_every == 0:
            validate_model(args, task_dir, tb_logger, valLoader, estimator, total_iter+1)

def calc_pos_cos_mask(pos_est, batch_size):
    # use max confidence point as sun pos est and calculate cosine mask
    pos_est_conf = pos_est.clone().detach().view(batch_size, -1) # B, 256
    max_idx = torch.argmax(pos_est_conf, dim=1) # B
    pos_est_conf[:, :] = 0.0
    pos_est_conf[torch.arange(batch_size), max_idx] = 1.0
    pos_est_conf = pos_est_conf.view(batch_size, 1, 8, 32) # B, 1, 8, 32
    idx_y, idx_x = np.unravel_index(max_idx.cpu().numpy(), (8, 32))
    azimuth_rad_est = (idx_x - 15.5)/16.0*np.pi # B
    elevation_rad_est = (7.5 - idx_y)/16.0*np.pi # B
    sun_unit_vec = np.array([np.cos(elevation_rad_est)*np.sin(azimuth_rad_est), # x
                            np.cos(elevation_rad_est)*np.cos(azimuth_rad_est), # y
                            np.sin(elevation_rad_est)]) # z 
    sun_unit_vec = sun_unit_vec.reshape(3, -1) # 3, B
    _tmp = np.mgrid[63:-1:-1,0:128:1]
    elevation_mask = _tmp[0][np.newaxis,:]
    azimuth_mask = _tmp[1][np.newaxis,:]
    elevation_mask = (elevation_mask - 31.5)/32*(np.pi/2) # 1, 64, 128
    azimuth_mask = (azimuth_mask - 63.5)/64*(np.pi) # 1, 64, 128
    unit_mask = np.stack([np.cos(elevation_mask)*np.sin(azimuth_mask),
                            np.cos(elevation_mask)*np.cos(azimuth_mask),
                            np.sin(elevation_mask)], axis=-1) # 1, 64, 128, 3
    cosine_mask_fine_est = -np.einsum('ijkl,lm->ijkm', unit_mask, sun_unit_vec) # 1, 64, 128, B
    cosine_mask_fine_est = np.clip(cosine_mask_fine_est, 0.0, 1.0).astype(np.float32)
    cosine_mask_fine_est = np.transpose(cosine_mask_fine_est, (3, 0, 1, 2)) # B, 1, 64, 128
    cosine_mask_fine_est = torch.tensor(cosine_mask_fine_est).to('cuda')
    sun_pos_y = (idx_y + 0.5) / 8.0 # B
    sun_pos_x = (idx_x + 0.5) / 32.0 # B
    sun_pos_y = np.clip(sun_pos_y, 0, 1)
    sun_pos_x = np.clip(sun_pos_x, 0, 1)
    pos_est_fine = np.zeros((batch_size, 1, 32, 128), dtype=np.float32) # B, 1, 32, 128
    idx_y = np.clip(sun_pos_y*32, 0, 31.99).astype(int)
    idx_x = np.clip(sun_pos_x*128, 0, 127.99).astype(int)
    pos_left_ind = np.maximum(int(0), idx_x-3)
    pos_right_ind = np.minimum(int(128), pos_left_ind+8)
    pos_left_ind = pos_right_ind - 8
    pos_upper_ind = np.maximum(int(0), idx_y-3)
    pos_lower_ind = np.minimum(int(32), pos_upper_ind+8)
    pos_upper_ind = pos_lower_ind - 8
    for _i in range(batch_size):
        pos_est_fine[_i, 0, pos_upper_ind[_i]:pos_lower_ind[_i], pos_left_ind[_i]:pos_right_ind[_i]] = 1.0
    pos_est_fine = torch.tensor(pos_est_fine).to('cuda')

    azimuth_deg = azimuth_rad_est / np.pi * 180.0
    elevation_deg = elevation_rad_est / np.pi * 180.0

    return (azimuth_deg, elevation_deg, pos_est_fine, cosine_mask_fine_est)

def save_model(args, task_dir, estimator, optimizer, scheduler, epoch=None, iter=None):
    os.makedirs(os.path.join(task_dir, 'checkpoints'), exist_ok=True)
    if epoch is not None:
        surfix = 'epoch_%d' %(epoch)
    if iter is not None:
        surfix = 'iter_%d' %(iter)
    if args.multi_gpu:
        torch.save(estimator.module.state_dict(), os.path.join(task_dir, 'checkpoints', 'estimator_latest'))
        torch.save(estimator.module.state_dict(), os.path.join(task_dir, 'checkpoints', 'estimator_%s' %(surfix)))
    else:
        torch.save(estimator.state_dict(), os.path.join(task_dir, 'checkpoints', 'estimator_latest'))
        torch.save(estimator.state_dict(), os.path.join(task_dir, 'checkpoints', 'estimator_%s' %(surfix)))
    torch.save(optimizer.state_dict(), os.path.join(task_dir, 'checkpoints', 'optimizer_latest'))
    torch.save(optimizer.state_dict(), os.path.join(task_dir, 'checkpoints', 'optimizer_%s' %(surfix)))
    torch.save(scheduler.state_dict(), os.path.join(task_dir, 'checkpoints', 'scheduler_latest'))
    torch.save(scheduler.state_dict(), os.path.join(task_dir, 'checkpoints', 'scheduler_%s' %(surfix)))

    print('-------------------------------------------------')
    print('            model saved at %s' %(surfix))
    print('-------------------------------------------------')

def validate_model(args, task_dir, tb_logger, valLoader, estimator, iter):
    print('-------------------------------------------------')
    print('            eval at %d iterations' %(iter))
    print('-------------------------------------------------')

    os.makedirs(os.path.join(task_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter)), exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'hdr'), exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'ldr'), exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'pano'), exist_ok=True)

    if args.multi_gpu:
        estimator.module.feat_exactor.eval()
        estimator.module.mask_decoder.eval()
        estimator.module.pos_decoder.eval()
        estimator.module.sky_estimator.eval()
        estimator.module.sun_estimator.eval()
        estimator.module.local_extractor.eval()
        estimator.module.local_estimator.eval()
    else:
        estimator.feat_exactor.eval()
        estimator.mask_decoder.eval()
        estimator.pos_decoder.eval()
        estimator.sky_estimator.eval()
        estimator.sun_estimator.eval()
        estimator.local_extractor.eval()
        estimator.local_estimator.eval()

    vgmse = []
    vgmae = []
    vgssim = []
    vlmae = []
    vlmse = []
    vlssim = []
    vaz = []
    vel = []

    vmmse = []
    vpbce = []
    vkcmse = []
    vscmse = []
    vlcmse = []
    vkmae = []
    vsmae = []
    vlamae = []
    vlsmse = []

    with torch.no_grad():
        total_vals = 0
        num_vis = 0
        for i, val_data in enumerate(tqdm(valLoader)):
            # retrieve the data
            persp_tensor = val_data['color'].to('cuda') # B, 3, 240, 320
            global_tensor = val_data['global_lighting'].to('cuda') # B, 3, 32, 128
            global_tensor.clamp_min_(0.0)
            global_tensor.clamp_max_(2000.0)
            global_sky_code = val_data['global_sky_code'].to('cuda') # B, 16
            global_sun_code = val_data['global_sun_code'].to('cuda') # B, 45
            local_pano = val_data['local_pano'].to('cuda') # B, 3, 64, 128
            local_pano.clamp_min_(0.0)
            local_pano.clamp_max_(2000.0)
            local_tensor = local_pano.clamp_max(1.0) # NOTE: for local component is OK
            local_mask = val_data['local_mask'].to('cuda') # B, 4, 1, 64, 128
            local_pos = val_data['local_pos'].to('cuda') # B, 4, 2
            local_code = val_data['local_code'].to('cuda') # B, 4, 64
            cosine_mask_fine = val_data['cosine_mask_fine'].to('cuda') # B, 1, 64, 128
            sun_vis = val_data['is_sunny'].to('cuda') # B
            sun_azimuth = val_data['sun_azimuth'] # B
            sun_elevation = val_data['sun_elevation'] # B
            sun_pos_mask = val_data['sun_pos_mask'].to('cuda') # B, 1, 8, 32
            sun_pos_mask_fine = val_data['sun_pos_mask_fine'].to('cuda') # B, 1, 32, 128
            persp_shadow_mask = val_data['persp_shadow_mask'].to('cuda') # B, 1, 240, 320

            local_pano_target_tensor = local_pano*(1-local_mask) + local_tensor*local_mask # NOTE: important here to bound local content less than 1.0

            # NOTE: use GT sky code, sun code and cosine mask here during training
            mask_est, pos_est, sky_code_est, sun_code_est, local_code_est, sky_est_raw, sun_est_raw, local_app_est_raw, local_sil_est, azimuth_deg_est, elevation_deg_est, pos_est_fine, cosine_mask_fine_est = estimator(persp_tensor, local_pos)

            B, num_local, D = local_pos.shape # B, num_local, 2

            # calcuate azimuth, elevation, pos mask and cosine mask predicts
            azimuth_deg_gt = sun_azimuth / np.pi * 180
            elevation_deg_gt = sun_elevation / np.pi * 180
            azimuth_deg_est = azimuth_deg_est.cpu()
            elevation_deg_est = elevation_deg_est.cpu()

            # decode codes to images
            sky_est = log2linear(sky_est_raw.clamp_min(0.0).clamp_max(4.5), args.log_mu) # B, 3, 32, 128
            sun_est = log2linear(sun_est_raw.clamp_min(0.0).clamp_max(4.5), args.log_mu) # B, 3, 32, 128
            
            local_app_est = log2linear(local_app_est_raw.clamp_min(0.0), args.log_mu) # B*num_local, 3, 64, 128
            local_app_est = local_app_est.view(B, num_local, 3, 64 ,128) # B, num_local, 3, 64, 128
            local_sil_est = local_sil_est.view(B, num_local, 1, 64 ,128) # B, num_local, 1, 64, 128

            # combine to the full local lighting map
            global_est = sky_est*(1-pos_est_fine) + sun_est # B, 3, 32, 128
            local_est = torch.nn.functional.pad(global_est, (0, 0, 0, 32)).unsqueeze(1)*(1-local_sil_est) + local_app_est*local_sil_est # B, 4, 3, 64, 128
            
            ldr_global = GammaTMO(global_tensor.cpu(), args.tmo_gamma, args.tmo_log_exposure)
            ldr_local = GammaTMO(local_pano_target_tensor.cpu(), args.tmo_gamma, args.tmo_log_exposure)

            ldr_global_est = GammaTMO(global_est.detach().cpu(), args.tmo_gamma, args.tmo_log_exposure)
            ldr_local_est = GammaTMO(local_est.detach().cpu(), args.tmo_gamma, args.tmo_log_exposure)

            # val loss stats
            for im_idx in range(B):
                vgmse.append(mse(global_est[im_idx], global_tensor[im_idx]).item())
                vgmae.append(mae(global_est[im_idx], global_tensor[im_idx]).item())
                vgssim.append(ssim(ldr_global_est[im_idx:im_idx+1], ldr_global[im_idx:im_idx+1]).item())
                for local_idx in range(num_local):
                    vlmse.append(mse(local_est[im_idx][local_idx], local_pano_target_tensor[im_idx][local_idx]).item())
                    vlmae.append(mae(local_est[im_idx][local_idx], local_pano_target_tensor[im_idx][local_idx]).item())
                    vlssim.append(ssim(ldr_local_est[im_idx][local_idx:local_idx+1], ldr_local[im_idx][local_idx:local_idx+1]).item())
                    vlcmse.append(mse(local_code_est[im_idx*num_local+local_idx], local_code[im_idx][local_idx]).item())
                if sun_vis[im_idx] == 1:
                    vaz.append(calc_azimuth_error(azimuth_deg_est[im_idx], azimuth_deg_gt[im_idx], unit='deg'))
                    vel.append(elevation_deg_est[im_idx] - elevation_deg_gt[im_idx])
                    vpbce.append(F.binary_cross_entropy(pos_est[im_idx].view(1, -1), sun_pos_mask[im_idx].view(1, -1)).item())
                vmmse.append(mse(mask_est[im_idx], persp_shadow_mask[im_idx]).item())
                vkcmse.append(mse(sky_code_est[im_idx], global_sky_code[im_idx]).item())
                vscmse.append(mse(sun_code_est[im_idx], global_sun_code[im_idx]).item())
                vkmae.append(mae(sky_est[im_idx], global_tensor[im_idx].clamp_max(1.0)).item())
                vsmae.append(mae(sun_est[im_idx], global_tensor[im_idx]*sun_pos_mask_fine[im_idx]).item())
                vlamae.append(mae(local_app_est[im_idx]*local_mask[im_idx], local_tensor[im_idx]*local_mask[im_idx]).item())
                vlsmse.append(mse(local_sil_est[im_idx], local_mask[im_idx]).item())

            for _i in range(B):
                total_vals += 1
                if total_vals%args.tb_save_image_every==0:
                    meta_data = '_'.join(val_data['meta'][_i].strip().split())
                    persp_sample = np.transpose(persp_tensor.cpu().numpy()[_i], (1, 2, 0)) # 240, 320, 3
                    persp_shadow_sample = np.transpose(persp_shadow_mask.cpu().numpy()[_i], (1, 2, 0)) # 240, 320, 1
                    persp_shadow_sample = np.repeat(persp_shadow_sample, 3, axis=2) # 240, 320, 3
                    local_pos_sample = local_pos[_i].cpu().numpy()
                    persp_loc_vis_sample = cv2.cvtColor((persp_sample*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    for local_idx in range(4):
                        if local_pos_sample[local_idx][0] < 0 or local_pos_sample[local_idx][1] < 0:
                            local_pos_sample[local_idx] = -local_pos_sample[local_idx] - 1
                        cv2.circle(persp_loc_vis_sample, (int(local_pos_sample[local_idx][1]), int(local_pos_sample[local_idx][0])), 1, colors[local_idx], 4)
                    persp_loc_vis_sample = cv2.cvtColor(persp_loc_vis_sample, cv2.COLOR_BGR2RGB)/255.0 # 240, 320, 3
                    cos_mask_sample = cosine_mask_fine.cpu().numpy()[_i]
                    cos_mask_sample = np.repeat(cos_mask_sample, 3, axis=0)
                    cos_mask_sample = np.transpose(cos_mask_sample, (1, 2, 0)) # 64, 128, 3
                    sun_posmask_sample = sun_pos_mask_fine.cpu().numpy()[_i] # 1, 32, 128
                    sun_posmask_sample = np.repeat(sun_posmask_sample, 3, axis=0) # 3, 32, 128
                    sun_posmask_sample = np.transpose(sun_posmask_sample, (1, 2, 0)) # 32, 128, 3

                    cos_mask_est_sample = cosine_mask_fine_est.detach().cpu().numpy()[_i] # 1, 64, 128
                    cos_mask_est_sample = np.repeat(cos_mask_est_sample, 3, axis=0)
                    cos_mask_est_sample = np.transpose(cos_mask_est_sample, (1, 2, 0)) # 64, 128, 3
                    sun_posmask_est_sample = np.transpose(pos_est_fine.detach().cpu().numpy()[_i], (1, 2, 0)) # 32, 128, 1
                    sun_posmask_est_sample = np.repeat(sun_posmask_est_sample, 3, axis=2)

                    global_sample = np.transpose(ldr_global.cpu().numpy()[_i], (1, 2, 0)) # 32, 128, 3
                    local_sil_sample = np.transpose(local_mask.cpu().numpy()[_i], (0, 2, 3, 1)) # 4, 64, 128, 1
                    local_sil_sample = np.repeat(local_sil_sample, 3, axis=3) # 4, 64, 128, 3
                    local_sample = np.transpose(ldr_local.cpu().numpy()[_i], (0, 2, 3, 1)) # 4, 64, 128, 3
                    local_sample_stitched = np.concatenate([
                        np.concatenate([local_sample[1], local_sample[0], local_sil_sample[1], local_sil_sample[0]], axis=1),
                        np.concatenate([local_sample[2], local_sample[3], local_sil_sample[2], local_sil_sample[3]], axis=1)
                    ], axis=0)
                    global_sample_stitched = np.concatenate([
                        np.concatenate([global_sample, sun_posmask_sample], axis=0),
                        cos_mask_sample
                    ], axis=0)

                    hdr_global_sample = np.transpose(global_tensor.cpu().numpy()[_i], (1, 2, 0)) # 32, 128, 3
                    hdr_local_sample = np.transpose(local_pano_target_tensor.cpu().numpy()[_i], (0, 2, 3, 1)) # 4, 64, 128, 3
                    hdr_local_sample_stitched = np.concatenate([
                        np.concatenate([hdr_local_sample[1], hdr_local_sample[0], local_sil_sample[1], local_sil_sample[0]], axis=1),
                        np.concatenate([hdr_local_sample[2], hdr_local_sample[3], local_sil_sample[2], local_sil_sample[3]], axis=1)
                    ], axis=0)
                    hdr_global_sample_stitched = np.concatenate([
                        np.concatenate([hdr_global_sample, sun_posmask_sample], axis=0),
                        cos_mask_sample
                    ], axis=0)

                    global_est_sample = np.transpose(ldr_global_est.detach().cpu().numpy()[_i], (1, 2, 0)) # 32, 128, 3
                    local_sil_est_sample = np.transpose(local_sil_est.detach().cpu().numpy()[_i], (0, 2, 3, 1)) # 4, 64, 128, 1
                    local_sil_est_sample = np.repeat(local_sil_est_sample, 3, axis=3) # 4, 64, 128, 3
                    local_est_sample = np.transpose(ldr_local_est.cpu().numpy()[_i], (0, 2, 3, 1)) # 4, 64, 128, 3

                    persp_shadow_est_sample = np.transpose(mask_est.detach().cpu().numpy()[_i], (1, 2, 0)) # 240, 320, 1
                    persp_shadow_est_sample = np.repeat(persp_shadow_est_sample, 3, axis=2) # 240, 320, 3

                    local_est_sample_stitched = np.concatenate([
                        np.concatenate([local_est_sample[1], local_est_sample[0], local_sil_est_sample[1], local_sil_est_sample[0]], axis=1),
                        np.concatenate([local_est_sample[2], local_est_sample[3], local_sil_est_sample[2], local_sil_est_sample[3]], axis=1)
                    ], axis=0)
                    global_est_sample_stitched = np.concatenate([
                        np.concatenate([global_est_sample, sun_posmask_est_sample], axis=0),
                        cos_mask_est_sample
                    ], axis=0)

                    hdr_global_est_sample = np.transpose(global_est.detach().cpu().numpy()[_i], (1, 2, 0)) # 32, 128, 3
                    hdr_local_est_sample = np.transpose(local_est.cpu().numpy()[_i], (0, 2, 3, 1)) # 4, 64, 128, 3

                    hdr_local_est_sample_stitched = np.concatenate([
                        np.concatenate([hdr_local_est_sample[1], hdr_local_est_sample[0], local_sil_est_sample[1], local_sil_est_sample[0]], axis=1),
                        np.concatenate([hdr_local_est_sample[2], hdr_local_est_sample[3], local_sil_est_sample[2], local_sil_est_sample[3]], axis=1)
                    ], axis=0)
                    hdr_global_est_sample_stitched = np.concatenate([
                        np.concatenate([hdr_global_est_sample, sun_posmask_est_sample], axis=0),
                        cos_mask_est_sample
                    ], axis=0)

                    pano_sample_stitched = np.concatenate([
                        np.concatenate([global_sample_stitched, local_sample_stitched], axis=1),
                        np.concatenate([global_est_sample_stitched, local_est_sample_stitched], axis=1)
                    ], axis=0)
                    persp_sample_stitched = np.concatenate([persp_loc_vis_sample, persp_shadow_sample, persp_shadow_est_sample], axis=1)
                    hdr_pano_sample_stitched = np.concatenate([
                        np.concatenate([hdr_global_sample_stitched, hdr_local_sample_stitched], axis=1),
                        np.concatenate([hdr_global_est_sample_stitched, hdr_local_est_sample_stitched], axis=1)
                    ], axis=0)


                    tb_logger.add_image("LDR Val sample %03d (pano)" %(num_vis+1), np.transpose(pano_sample_stitched, [2, 0 ,1]), iter)
                    tb_logger.add_image("LDR Val sample %03d (perspective)" %(num_vis+1), np.transpose(persp_sample_stitched, [2, 0, 1]), iter)
                    tb_logger.add_image("HDR Val sample %03d (pano)" %(num_vis+1), np.transpose(hdr_pano_sample_stitched, [2, 0 ,1]), iter)

                    imageio.imwrite(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'ldr', 'sample_%03d_%s_pano.png' %(num_vis+1, meta_data)), (pano_sample_stitched*255.0).astype(np.uint8), format='PNG')
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'ldr', 'sample_%03d_%s_perspective.png' %(num_vis+1, meta_data)), (persp_sample_stitched*255.0).astype(np.uint8), format='PNG')
                    imageio.imwrite(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'hdr', 'sample_%03d_%s_pano.hdr' %(num_vis+1, meta_data)), hdr_pano_sample_stitched, format='HDR-FI')

                    for _i in [1, 0, 2, 3]: # NOTE: l to r, u to b instead of quardant order
                        imageio.imwrite(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'pano', 'sample_%03d_%s_%d_est.hdr' %(num_vis+1, meta_data, _i)), hdr_local_sample[_i], format='HDR-FI')
                        imageio.imwrite(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'pano', 'sample_%03d_%s_%d_gt.hdr' %(num_vis+1, meta_data, _i)), hdr_local_est_sample[_i], format='HDR-FI')

                    num_vis += 1

        vgmse_array = np.array(vgmse)
        vgmae_array = np.array(vgmae)
        vgssim_array = np.array(vgssim)
        vlmae_array = np.array(vlmae)
        vlmse_array = np.array(vlmse)
        vlssim_array = np.array(vlssim)
        vaz_array = np.array(vaz)
        vel_array = np.array(vel)

        vmmse_array = np.array(vmmse)
        vpbce_array = np.array(vpbce)
        vkcmse_array = np.array(vkcmse)
        vscmse_array = np.array(vscmse)
        vlcmse_array = np.array(vlcmse)
        vkmae_array = np.array(vkmae)
        vsmae_array = np.array(vsmae)
        vlamae_array = np.array(vlamae)
        vlsmse_array = np.array(vlsmse)

        if tb_logger:
            tb_logger.add_scalar("Val Summary (G-MAE)", np.nanmean(vgmae_array), iter)
            tb_logger.add_scalar("Val Summary (G-MSE)", np.nanmean(vgmse_array), iter)
            tb_logger.add_scalar("Val Summary (G-RMSE)", np.nanmean(np.sqrt(vgmse_array)), iter)
            tb_logger.add_scalar("Val Summary (G-SSIM)", np.nanmean(vgssim_array), iter)
            tb_logger.add_scalar("Val Summary (L-MAE)", np.nanmean(vlmae_array), iter)
            tb_logger.add_scalar("Val Summary (L-MSE)", np.nanmean(vlmse_array), iter)
            tb_logger.add_scalar("Val Summary (L-RMSE)", np.nanmean(np.sqrt(vlmse_array)), iter)
            tb_logger.add_scalar("Val Summary (L-SSIM)", np.nanmean(vlssim_array), iter)
            tb_logger.add_scalar("Val Summary (AZ-d)", np.nanmean(abs(vaz_array)), iter)
            tb_logger.add_scalar("Val Summary (EL-d)", np.nanmean(abs(vel_array)), iter)
            add_tf_summary_histogram(tb_logger, "Val azimuth_error(deg)", vaz_array, iter)
            add_tf_summary_histogram(tb_logger, "Val elevation_error(deg)", vel_array, iter)

            tb_logger.add_scalar("Shadow Loss (MSE-V)", np.nanmean(vmmse_array), iter)
            tb_logger.add_scalar("Pos Loss (BCE-V)", np.nanmean(vpbce_array), iter)
            tb_logger.add_scalar("Sky Code Loss (MSE-V)", np.nanmean(vkcmse_array), iter)
            tb_logger.add_scalar("Sun Code Loss (MSE-V)", np.nanmean(vscmse_array), iter)
            tb_logger.add_scalar("Local Code Loss (MSE-V)", np.nanmean(vlcmse_array), iter)
            tb_logger.add_scalar("Sky Loss (MAE-V)", np.nanmean(vkmae_array), iter)
            tb_logger.add_scalar("Sun Loss (MAE-V)", np.nanmean(vsmae_array), iter)
            tb_logger.add_scalar("Local App Loss (MAE-V)", np.nanmean(vlamae_array), iter)
            tb_logger.add_scalar("Local Sil Loss (MSE-V)", np.nanmean(vlsmse_array), iter)

        val_losses = {
            "vgmse_array": vgmse_array,
            "vgmae_array": vgmae_array,
            "vgssim_array": vgssim_array,
            "vlmae_array": vlmae_array,
            "vlmse_array": vlmse_array,
            "vlssim_array": vlssim_array,
            "vaz_array": vaz_array,
            "vel_array": vel_array,
            "vmmse_array": vmmse_array,
            "vpbce_array": vpbce_array,
            "vkcmse_array": vkcmse_array,
            "vscmse_array": vscmse_array,
            "vlcmse_array": vlcmse_array,
            "vkmae_array": vkmae_array,
            "vsmae_array": vsmae_array,
            "vlamae_array": vlamae_array,
            "vlsmse_array": vlsmse_array
        }
        
        os.makedirs(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'losses'), exist_ok=True)
        with open(os.path.join(task_dir, 'samples', 'iter_%08d' %(iter), 'losses', 'all_losses.pkl'), 'wb') as f:
            pickle.dump(val_losses, f)

        print("Stats: G-MAE = %f, G-MSE = %f, G-RMSE = %f, G-SSIM = %f" %(np.nanmean(vgmae_array), np.nanmean(vgmse_array), np.nanmean(np.sqrt(vgmse_array)), np.nanmean(vgssim_array)))
        print("Stats: L-MAE = %f, L-MSE = %f, L-RMSE = %f, L-SSIM = %f" %(np.nanmean(vlmae_array), np.nanmean(vlmse_array), np.nanmean(np.sqrt(vlmse_array)), np.nanmean(vlssim_array)))
        print("Stats: AZ-d = %f, EL-d = %f" %(np.nanmean(abs(vaz_array)), np.nanmean(abs(vel_array))))

    if args.multi_gpu:
        estimator.module.feat_exactor.train()
        estimator.module.mask_decoder.train()
        estimator.module.pos_decoder.train()
        estimator.module.sky_estimator.train()
        estimator.module.sun_estimator.train()
        estimator.module.local_extractor.train()
        estimator.module.local_estimator.train()
    else:
        estimator.feat_exactor.train()
        estimator.mask_decoder.train()
        estimator.pos_decoder.train()
        estimator.sky_estimator.train()
        estimator.sun_estimator.train()
        estimator.local_extractor.train()
        estimator.local_estimator.train()

    print('-------------------------------------------------')
    print('            eval finish')
    print('-------------------------------------------------')

if __name__ == '__main__':
    main()