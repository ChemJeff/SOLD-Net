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

from model.Autoencoder import LocalEncoder

class LocalLightingDset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot
        city_cams = sorted([path.split('/')[-1] for path in glob.glob(os.path.join(dataroot, 'local_pano', 'City_*_camera_*'))])
        self.items = self.get_item_lists(city_cams, dataroot)
        print("num_items: %d" %(len(self.items)))

    def get_item_lists(self, cam_list, dataroot):
        lists = []
        for cam in cam_list:
            print(os.path.join(dataroot, 'local_pano', cam))
            skies = os.listdir(os.path.join(dataroot, 'local_pano', cam, 'rendered'))
            for sky in skies:
                for ang in range(6):
                    for lcam in range(4):
                        if not os.path.exists(os.path.join(dataroot, 'local_pano', cam, 'rendered', sky, 'angle_%d_camera_%d.hdr' %(ang, lcam))):
                            continue
                        lists.append([cam, sky, ang, lcam])
        return lists

    def get_item(self, idx):
        item = self.items[idx]
        city_cam_name, sky_name, angle_id, lcam = item
        hdr_path = os.path.join(self.dataroot, 'local_pano', city_cam_name, 'rendered', sky_name, 'angle_%d_camera_%d.hdr' %(angle_id, lcam))
        hdr = imageio.imread(hdr_path, format='HDR-FI') # H * W * C
        hdr = np.transpose(np.asfarray(hdr, dtype=np.float32), (2, 0, 1)) # C * H * W
        return_dict = {
            'color': hdr,
            'city_cam_name': city_cam_name,
            'sky_name': sky_name,
            'angle_id': str(angle_id),
            'lcam': str(lcam)
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
parser.add_argument("--load_local_enc_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
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
enc_local = LocalEncoder(cin=3, cout=64, activ=args.model_activ).to('cuda')

print("output path: ", output_dir)

# load checkpoints
print('loading local encoder from ', args.load_local_enc_path)
enc_local.load_state_dict(torch.load(args.load_local_enc_path, map_location='cuda'))

dataset = LocalLightingDset(dataroot)
dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_loader, drop_last=False)

enc_local.eval()

all_data = {}

with torch.no_grad():
    for i, test_data in tqdm(enumerate(dataLoader)):
        local_tensor = test_data['color'].to('cuda')
        local_tensor = local_tensor*4.0
        local_tensor.clamp_min_(0.0)
        local_tensor.clamp_max_(1.0) # NOTE: for local component is OK
        city_cam_name = test_data['city_cam_name']
        sky_name = test_data['sky_name']
        angle_id = test_data['angle_id']
        lcam = test_data['lcam']

        local_target_tensor = local_tensor

        if args.log_image:
            local_input_tensor = recon_local_target_tensor = linear2log(local_tensor, args.log_mu)
        else:
            local_input_tensor = recon_local_target_tensor = local_tensor

        latent_local = enc_local(local_input_tensor)

        local_code = latent_local.detach().cpu().numpy()

        for idx in range(latent_local.shape[0]):
            _city_cam = city_cam_name[idx]
            _sky = sky_name[idx]
            _ang = angle_id[idx]
            _lcam = lcam[idx]
            basepath = os.path.join(output_dir, _city_cam, 'rendered', _sky)
            os.makedirs(basepath, exist_ok=True)
            local_code_path = os.path.join(basepath, 'angle_%s_local_code_%s.npy' %(_ang, _lcam))
            with open(local_code_path, 'wb') as f:
                np.save(f, local_code[idx])
            
            all_data[_city_cam] = all_data.get(_city_cam, {})
            all_data[_city_cam][_sky] = all_data[_city_cam].get(_sky, {})
            all_data[_city_cam][_sky][_ang] = all_data[_city_cam][_sky].get(_ang, {})
            all_data[_city_cam][_sky][_ang][_lcam] = {
                'local': local_code[idx]
            }

with open(os.path.join(output_dir, 'all_dump.pkl'), 'wb') as f:
    pickle.dump(all_data, f)

print()
print("All done. Codes have been saved to %s" %(output_dir))
