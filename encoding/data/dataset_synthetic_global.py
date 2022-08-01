import os
import numpy as np
import imageio
import pickle
from torch.utils.data import Dataset

class SynGlobalDataset(Dataset):
    def __init__(self, opt, dataroot, splitfile, phase="train", filterfile=None):
        assert phase.lower() in ["train", "val", "test"]
        self.opt = opt
        self.phase = phase
        self.dataroot = dataroot
        with open(splitfile, 'r') as f:
            self.items = [line.strip().split() for line in f.readlines() if line.strip() and "noon_grass" not in line.strip().split()] # NOTE: manually remove outliers
        if filterfile is not None:
            if isinstance(filterfile, str):
                with open(filterfile, 'r') as f:
                    self.filtered_items = [line.strip().split() for line in f.readlines() if line.strip()]
            else:
                assert isinstance(filterfile, list)
                self.filtered_items = []
                for file in filterfile:
                    with open(file, 'r') as f:
                        self.filtered_items.extend([line.strip().split() for line in f.readlines() if line.strip()])
            print("found %d filter items, now filtering" %(len(self.filtered_items)))
            for item in self.items[:]: # NOTE: if not iterating on the new list ([:]), the filtering is incomplete!
                if item in self.filtered_items:
                    self.items.remove(item)
        print("dataset phase: %s" %(phase))
        print("num items: %d" %(len(self.items)))

    def get_item(self, idx):
        item = self.items[idx]
        city_cam_name, sky_name, angle_id = item
        hdr_path = os.path.join(self.dataroot, 'global_lighting', city_cam_name, 'rendered', sky_name, 'global_lighting_%s.hdr' %(angle_id))
        hdr = imageio.imread(hdr_path, format='HDR-FI') # H * W * C
        hdr = np.transpose(np.asfarray(hdr, dtype=np.float32), (2, 0, 1)) # C * H * W
        Ci, Hi, Wi = hdr.shape
        return_dict = {
            'color': hdr,
            'color_path': hdr_path,
            'city_cam_name': city_cam_name,
            'sky_name': sky_name,
            'angle_id': angle_id
        }
        sun_param_path = os.path.join(self.dataroot, 'global_lighting', city_cam_name, 'rendered', sky_name, 'sun_pos_%s.pkl' %(angle_id))
        with open(sun_param_path, 'rb') as f:
            sun_params = pickle.load(f)
        if not hasattr(self.opt, 'sunmask_scale'):
            self.sunmask_scale = 4
        else:
            self.sunmask_scale = self.opt.sunmask_scale
        pos_mask = np.zeros((1, Hi//self.sunmask_scale, Wi//self.sunmask_scale), dtype=np.float32)
        pos_mask_coarse = np.zeros((1, Hi//(2*self.sunmask_scale), Wi//(2*self.sunmask_scale)), dtype=np.float32)
        pos_mask_fine = np.zeros((1, Hi, Wi), dtype=np.float32)
        sun_pos_y = 1 - sun_params['elevation']/(np.pi/2)
        sun_pos_y = np.clip(sun_pos_y, 0, 1)
        sun_pos_x = 0.5 + sun_params['azimuth']/(np.pi*2)
        sun_pos_x = np.clip(sun_pos_x, 0, 1)
        if sun_params['is_sunny']: # NOTE: if cloudy, set sunpos mask all zeros!
            idx_y = int(np.clip(sun_pos_y*8, 0, 7.99))
            idx_x = int(np.clip(sun_pos_x*32, 0, 31.99))
            pos_mask[0][idx_y][idx_x] = 1.0
            idx_y = int(np.clip(sun_pos_y*4, 0, 3.99))
            idx_x = int(np.clip(sun_pos_x*16, 0, 15.99))
            pos_mask_coarse[0][idx_y][idx_x] = 1.0
            idx_y = int(np.clip(sun_pos_y*32, 0, 31.99))
            idx_x = int(np.clip(sun_pos_x*128, 0, 127.99))
            pos_left_ind = max(int(0), idx_x-3)
            pos_right_ind = min(int(128), pos_left_ind+8)
            pos_left_ind = pos_right_ind - 8
            pos_upper_ind = max(int(0), idx_y-3)
            pos_lower_ind = min(int(32), pos_upper_ind+8)
            pos_upper_ind = pos_lower_ind - 8
            pos_mask_fine[0, pos_upper_ind:pos_lower_ind, pos_left_ind:pos_right_ind] = 1.0
        return_dict['sun_pos_mask'] = pos_mask
        return_dict['sun_pos_mask_coarse'] = pos_mask_coarse
        return_dict['sun_pos_mask_fine'] = pos_mask_fine
        return_dict['sun_vis'] = 0.0 if sun_params['is_sunny'] is False else 1.0
        return_dict['sun_x'] = sun_pos_x
        return_dict['sun_y'] = sun_pos_y
        return return_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)
