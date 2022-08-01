import os
import numpy as np
import imageio
import pickle
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, opt, dataroot, splitfile, phase="train", filterfile=None):
        assert phase.lower() in ["train", "val", "test"]
        self.opt = opt
        self.phase = phase
        self.dataroot = dataroot
        with open(splitfile, 'r') as f:
            self.items = [line.strip().split() for line in f.readlines() if line.strip()]
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
        with open(os.path.join(self.dataroot, 'global_lighting_code', 'all_dump.pkl'), 'rb') as f:
            self.global_codes = pickle.load(f)
        with open(os.path.join(self.dataroot, 'local_lighting_code', 'all_dump.pkl'), 'rb') as f:
            self.local_codes = pickle.load(f)
        print("dataset phase: %s" %(phase))
        print("num items: %d" %(len(self.items)))

    def get_item(self, idx):
        item = self.items[idx]
        city_cam_name, sky_name, angle_id = item
        persp_ldr_path = os.path.join(self.dataroot, 'perspective', city_cam_name, 'rendered', sky_name, 'image_%s.png' %(angle_id))
        persp_shadow_mask_path = os.path.join(self.dataroot, 'sun_invisible_mask', city_cam_name, 'rendered', sky_name, 'sun_invisible_mask_%s.png' %(angle_id))
        global_light_stat_path = os.path.join(self.dataroot, 'global_lighting', city_cam_name, 'rendered', sky_name, 'sun_pos_%s.pkl' %(angle_id))
        global_light_path = os.path.join(self.dataroot, 'global_lighting', city_cam_name, 'rendered', sky_name, 'global_lighting_%s.hdr' %(angle_id))
        persp = imageio.imread(persp_ldr_path, format='PNG') # H * W * C
        persp = np.transpose(np.asfarray(persp, dtype=np.float32), (2, 0, 1)) / 255.0 # C * H * W
        persp_shadow_mask = imageio.imread(persp_shadow_mask_path, format='PNG') # H * W
        persp_shadow_mask = np.asfarray(persp_shadow_mask, dtype=np.float32) / 255.0 # H * W
        global_light_hdr = imageio.imread(global_light_path, format='HDR-FI')*4.0 # NOTE: fix scale
        global_light_hdr = np.transpose(np.asfarray(global_light_hdr, dtype=np.float32), (2, 0, 1))
        with open(global_light_stat_path, 'rb') as f:
            global_light = pickle.load(f)
        global_sky_code = self.global_codes[city_cam_name][sky_name][angle_id]['sky']
        global_sun_code = self.global_codes[city_cam_name][sky_name][angle_id]['sun']
        local_panos = []
        local_masks = []
        local_codes = []
        for local_id in range(4):
            _local_pano_path = os.path.join(self.dataroot, 'local_pano', city_cam_name, 'rendered', sky_name, 'angle_%s_camera_%d.hdr' %(angle_id, local_id))
            _local_mask_path = os.path.join(self.dataroot, 'local_pano', city_cam_name, 'rendered', sky_name, 'angle_%s_camera_%d_mask.png' %(angle_id, local_id))
            _local_pano = imageio.imread(_local_pano_path, format='HDR-FI')*4.0 # NOTE: fix scale
            _local_pano = np.transpose(np.asfarray(_local_pano, dtype=np.float32), (2, 0, 1))
            _local_mask = imageio.imread(_local_mask_path, format='PNG')
            _local_mask = np.asfarray(_local_mask, dtype=np.float32) / 255.0
            local_panos.append(_local_pano)
            local_masks.append(_local_mask[np.newaxis, :, :])
            local_codes.append(self.local_codes[city_cam_name][sky_name][angle_id][str(local_id)]['local'])
        local_panos = np.stack(local_panos, axis=0)
        local_masks = np.stack(local_masks, axis=0)
        local_codes = np.stack(local_codes, axis=0)
        local_pos_path = os.path.join(self.dataroot, 'local_cameras', city_cam_name, 'local_samples', sky_name, 'image_%s_local.npy' %(angle_id))
        local_pos = np.load(local_pos_path)
        for local_id in range(4):
            # NOTE: modified read pixl_index
            if local_pos[local_id][0] < 0 or local_pos[local_id][1] < 0:
                    local_pos[local_id] = -local_pos[local_id] - 1
        return_dict = {
            'is_sunny': np.float32(1.0) if global_light['is_sunny'] else np.float32(0.0),
            'sun_elevation': np.float32(global_light['elevation']),
            'sun_azimuth': np.float32(global_light['azimuth']),
            'color': persp,
            'global_lighting': global_light_hdr,
            'global_sky_code': global_sky_code,
            'global_sun_code': global_sun_code,
            'local_pos': local_pos,
            'local_mask': local_masks,
            'local_pano': local_panos,
            'local_code': local_codes,
            'persp_shadow_mask': persp_shadow_mask[np.newaxis, :, :],
            'meta': ' '.join([city_cam_name, sky_name, angle_id])
        }
        sun_pos_mask = np.zeros((1, 8, 32), dtype=np.float32)
        pos_mask_fine = np.zeros((1, 32, 128), dtype=np.float32)
        sun_pos_y = 1 - global_light['elevation']/(np.pi/2)
        sun_pos_y = np.clip(sun_pos_y, 0, 1)
        sun_pos_x = 0.5 + global_light['azimuth']/(np.pi*2)
        sun_pos_x = np.clip(sun_pos_x, 0, 1)
        if global_light['is_sunny']: # NOTE: if cloudy, set sunpos mask all zeros!
            _tmp = np.mgrid[63:-1:-1,0:128:1]
            elevation_mask = _tmp[0]
            azimuth_mask = _tmp[1]
            elevation_mask = (elevation_mask - 31.5)/32*(np.pi/2) # 64, 128
            azimuth_mask = (azimuth_mask - 63.5)/64*(np.pi) # 64, 128
            sun_unit_vec = np.array([np.cos(global_light['elevation'])*np.sin(global_light['azimuth']), # x
                                    np.cos(global_light['elevation'])*np.cos(global_light['azimuth']), # y
                                    np.sin(global_light['elevation'])]) # z
            unit_mask = np.stack([np.cos(elevation_mask)*np.sin(azimuth_mask),
                                   np.cos(elevation_mask)*np.cos(azimuth_mask),
                                   np.sin(elevation_mask)], axis=-1) # 64, 128, 3
            cosine_mask_fine = -np.einsum('ijk,k->ij', unit_mask, sun_unit_vec)[np.newaxis, :, :] # 1, 64, 128
            cosine_mask_fine = np.clip(cosine_mask_fine, 0.0, 1.0).astype(np.float32)
            idx_y = int(np.clip(sun_pos_y*8, 0, 7.99))
            idx_x = int(np.clip(sun_pos_x*32, 0, 31.99))
            sun_pos_mask[0][idx_y][idx_x] = 1.0
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
        return_dict['cosine_mask_fine'] = cosine_mask_fine
        return_dict['sun_pos_mask_fine'] = pos_mask_fine
        return_dict['sun_pos_mask'] = sun_pos_mask
        return_dict['weather'] = np.array([1.0, 0.0], dtype=np.float32) if global_light['is_sunny'] else np.array([0.0, 1.0], dtype=np.float32)
        return_dict['sun_pos_y'] = sun_pos_y
        return_dict['sun_pos_x'] = sun_pos_x
        return return_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)