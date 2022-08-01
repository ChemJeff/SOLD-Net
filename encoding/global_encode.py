import os
import glob
import argparse
import pickle
import torch
import imageio
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataset
from tqdm import tqdm
from utils.mapping.log_mapping import linear2log, log2linear
from utils.logger import *

from model.Autoencoder import GlobalEncoder

class GlobalLightingDset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot
        city_cams = sorted([path.split('/')[-1] for path in glob.glob(os.path.join(dataroot, 'global_lighting', 'City_*_camera_*'))])
        self.items = self.get_item_lists(city_cams, dataroot)
        print("num_items: %d" %(len(self.items)))

    def get_item_lists(self, cam_list, dataroot):
        lists = []
        for cam in cam_list:
            print(os.path.join(dataroot, 'global_lighting', cam))
            skies = os.listdir(os.path.join(dataroot, 'global_lighting', cam, 'rendered'))
            for sky in skies:
                for ang in range(6):
                    if not os.path.exists(os.path.join(dataroot, 'global_lighting', cam, 'rendered', sky, 'global_lighting_%d.hdr' %(ang))):
                        continue
                    lists.append([cam, sky, ang])
        return lists

    def get_item(self, idx):
        item = self.items[idx]
        city_cam_name, sky_name, angle_id = item
        hdr_path = os.path.join(self.dataroot, 'global_lighting', city_cam_name, 'rendered', sky_name, 'global_lighting_%s.hdr' %(angle_id))
        hdr = imageio.imread(hdr_path, format='HDR-FI') # H * W * C
        hdr = np.transpose(np.asfarray(hdr, dtype=np.float32), (2, 0, 1)) # C * H * W
        return_dict = {
            'color': hdr,
            'city_cam_name': city_cam_name,
            'sky_name': sky_name,
            'angle_id': str(angle_id)
        }
        return return_dict
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--override", action="store_true", help="debug mode")
parser.add_argument("--dataroot_syn", type=str, default="../data/synthetic/", help="root of synthetic dataset")
parser.add_argument("--log_image", action="store_true", help="use image in log space")
parser.add_argument("--log_mu", type=float, default=16.0)
parser.add_argument("--load_sky_enc_path", type=str, required=True)
parser.add_argument("--load_sun_enc_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--sky_dim", type=int, default=16)
parser.add_argument("--sun_dim", type=int, default=45)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_loader", type=int, default=8)
parser.add_argument("--model_activ", type=str, choices=['relu', 'lrelu'], default='relu')

args = parser.parse_args()

dataroot = args.dataroot_syn
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=(args.debug or args.override))

save_options_cmdline(output_dir, args)
logger = set_logger(output_dir)
tb_logger = set_tb_logger(output_dir)
tb_save_options_cmdline(tb_logger, args)

# initialize models
enc_sky = GlobalEncoder(cin=3, cout=args.sky_dim, activ=args.model_activ).to('cuda')
enc_sun = GlobalEncoder(cin=3, cout=args.sun_dim, activ=args.model_activ).to('cuda')

print("output path: ", output_dir)

# load checkpoints
print('loading sky encoder from ', args.load_sky_enc_path)
enc_sky.load_state_dict(torch.load(args.load_sky_enc_path, map_location='cuda'))
print('loading sun encoder from ', args.load_sun_enc_path)
enc_sun.load_state_dict(torch.load(args.load_sun_enc_path, map_location='cuda'))

dataset = GlobalLightingDset(dataroot)
dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_loader, drop_last=False)

enc_sky.eval()
enc_sun.eval()

all_data = {}

with torch.no_grad():
    for i, test_data in tqdm(enumerate(dataLoader)):
        image_tensor = test_data['color'].to('cuda')
        image_tensor = image_tensor*4.0
        image_tensor.clamp_min_(0.0)
        image_tensor.clamp_max_(2000.0)
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

        sky_code = latent_sky.detach().cpu().numpy()
        sun_code = latent_sun.detach().cpu().numpy()

        for idx in range(latent_sky.shape[0]):
            _city_cam = city_cam_name[idx]
            _sky = sky_name[idx]
            _ang = angle_id[idx]
            basepath = os.path.join(output_dir, _city_cam, 'rendered', _sky)
            os.makedirs(basepath, exist_ok=True)
            sky_code_path = os.path.join(basepath, 'sky_code_%s.npy' %(_ang))
            sun_code_path = os.path.join(basepath, 'sun_code_%s.npy' %(_ang))
            with open(sky_code_path, 'wb') as f:
                np.save(f, sky_code[idx])
            with open(sun_code_path, 'wb') as f:
                np.save(f, sun_code[idx])
            
            all_data[_city_cam] = all_data.get(_city_cam, {})
            all_data[_city_cam][_sky] = all_data[_city_cam].get(_sky, {})
            all_data[_city_cam][_sky][_ang] = {
                'sky': sky_code[idx],
                'sun': sun_code[idx]
            }

with open(os.path.join(output_dir, 'all_dump.pkl'), 'wb') as f:
    pickle.dump(all_data, f)

print()
print("All done. Codes have been saved to %s" %(output_dir))
