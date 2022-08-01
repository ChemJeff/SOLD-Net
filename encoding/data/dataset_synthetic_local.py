import os
import numpy as np
import imageio
import pickle
from torch.utils.data import Dataset

class SynLocalDataset(Dataset):
    def __init__(self, opt, dataroot, splitfile, phase="train", filterfile=None):
        assert phase.lower() in ["train", "val", "test"]
        self.opt = opt
        self.phase = phase
        self.dataroot = dataroot
        with open(splitfile, 'r') as f:
            self.items_global = [line.strip().split() for line in f.readlines() if line.strip() and "noon_grass" not in line.strip().split()]
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
            for item in self.items_global[:]: # NOTE: if not iterating on the new list ([:]), the filtering is incomplete!
                if item in self.filtered_items:
                    self.items_global.remove(item)
        self.items = []
        for global_item in self.items_global:
            city_cam_name, sky_name, angle_id = global_item
            for local_id in range(4):
                self.items.append([city_cam_name, sky_name, angle_id, str(local_id)])
        print("dataset phase: %s" %(phase))
        print("num items: %d" %(len(self.items)))

    def get_item(self, idx):
        item = self.items[idx]
        city_cam_name, sky_name, angle_id, local_id = item
        global_hdr_path = os.path.join(self.dataroot, 'global_lighting', city_cam_name, 'rendered', sky_name, 'global_lighting_%s.hdr' %(angle_id))
        global_hdr = imageio.imread(global_hdr_path, format='HDR-FI') # H * W * C
        global_hdr = np.transpose(np.asfarray(global_hdr, dtype=np.float32), (2, 0, 1)) # C * H * W
        local_hdr_path = os.path.join(self.dataroot, 'local_pano', city_cam_name, 'rendered', sky_name, 'angle_%s_camera_%s.hdr' %(angle_id, local_id))
        local_hdr = imageio.imread(local_hdr_path, format='HDR-FI') # H * W * C
        local_hdr = np.transpose(np.asfarray(local_hdr, dtype=np.float32), (2, 0, 1)) # C * H * W
        local_mask_path = os.path.join(self.dataroot, 'local_pano', city_cam_name, 'rendered', sky_name, 'angle_%s_camera_%s_mask.png' %(angle_id, local_id))
        local_mask = imageio.imread(local_mask_path, format='PNG') # H * W
        local_mask = np.asfarray(local_mask, dtype=np.float32)[np.newaxis, :, :] # C * H * W
        local_mask[local_mask > 0] = 1.0
        Ci, Hi, Wi = local_hdr.shape
        return_dict = {
            'global': global_hdr,
            'local': local_hdr,
            'local_mask': local_mask,
            'metadata': ' '.join([city_cam_name, sky_name, angle_id, local_id])
        }
        sun_param_path = os.path.join(self.dataroot, 'global_lighting', city_cam_name, 'rendered', sky_name, 'sun_pos_%s.pkl' %(angle_id))
        with open(sun_param_path, 'rb') as f:
            sun_params = pickle.load(f)
        if sun_params['is_sunny']: # NOTE: if cloudy, set sunpos mask all zeros!
            _tmp = np.mgrid[63:-1:-1,0:128:1]
            elevation_mask = _tmp[0]
            azimuth_mask = _tmp[1]
            elevation_mask = (elevation_mask - 31.5)/32*(np.pi/2) # 64, 128
            azimuth_mask = (azimuth_mask - 63.5)/64*(np.pi) # 64, 128
            sun_unit_vec = np.array([np.cos(sun_params['elevation'])*np.sin(sun_params['azimuth']), # x
                                    np.cos(sun_params['elevation'])*np.cos(sun_params['azimuth']), # y
                                    np.sin(sun_params['elevation'])]) # z
            unit_mask = np.stack([np.cos(elevation_mask)*np.sin(azimuth_mask),
                                   np.cos(elevation_mask)*np.cos(azimuth_mask),
                                   np.sin(elevation_mask)], axis=-1) # 64, 128, 3
            cosine_mask_fine = -np.einsum('ijk,k->ij', unit_mask, sun_unit_vec)[np.newaxis, :, :] # 1, 64, 128
            cosine_mask_fine = np.clip(cosine_mask_fine, 0.0, 1.0).astype(np.float32)
            pos_mask_fine = np.zeros((1, 64, 128), dtype=np.float32)
            sun_pos_y = 1 - sun_params['elevation']/(np.pi/2)
            sun_pos_y = np.clip(sun_pos_y, 0, 1)
            sun_pos_x = 0.5 + sun_params['azimuth']/(np.pi*2)
            sun_pos_x = np.clip(sun_pos_x, 0, 1)
            idx_y = int(np.clip(sun_pos_y*32, 0, 31.99))
            idx_x = int(np.clip(sun_pos_x*128, 0, 127.99))
            pos_left_ind = max(int(0), idx_x-3)
            pos_right_ind = min(int(128), pos_left_ind+8)
            pos_left_ind = pos_right_ind - 8
            pos_upper_ind = max(int(0), idx_y-3)
            pos_lower_ind = min(int(32), pos_upper_ind+8)
            pos_upper_ind = pos_lower_ind - 8
            pos_mask_fine[0, pos_upper_ind:pos_lower_ind, pos_left_ind:pos_right_ind] = 1.0
        else:
            cosine_mask_fine = np.zeros((1, 64, 128), dtype=np.float32)
            pos_mask_fine = np.zeros((1, 64, 128), dtype=np.float32)
        return_dict['cosine_mask_fine'] = cosine_mask_fine
        return_dict['sun_pos_mask_fine'] = pos_mask_fine
        return return_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)
