import os
import numpy as np
import imageio
import pickle
from torch.utils.data import Dataset


class MixedGlobalDataset(Dataset):
    def __init__(self, opt, dataroot_syn, splitfile_syn, dataroot_laval, splitfile_laval, dataroot_sun_laval, phase="train", filterfile_syn=None, filterfile_laval=None):
        assert phase.lower() in ["train", "val", "test"]
        self.opt = opt
        self.phase = phase
        self.dataroot_syn = dataroot_syn
        self.dataroot_laval = dataroot_laval
        self.dataroot_sun_laval = dataroot_sun_laval

        # load synthetic data
        with open(splitfile_syn, 'r') as f:
            self.items_syn = [line.strip().split() for line in f.readlines() if line.strip()]
        if filterfile_syn is not None:
            if isinstance(filterfile_syn, str):
                with open(filterfile_syn, 'r') as f:
                    self.filtered_items_syn = [line.strip().split() for line in f.readlines() if line.strip()]
            else:
                assert isinstance(filterfile_syn, list)
                self.filtered_items_syn = []
                for file in filterfile_syn:
                    with open(file, 'r') as f:
                        self.filtered_items_syn.extend([line.strip().split() for line in f.readlines() if line.strip()])
            print("found %d filter items (synthetic), now filtering" %(len(self.filtered_items_syn)))
            for item in self.items_syn[:]: # NOTE: if not iterating on the new list ([:]), the filtering is incomplete!
                if item in self.filtered_items_syn:
                    self.items_syn.remove(item)
        
        # load laval data 
        with open(splitfile_laval, 'r') as f:
            self.items_laval = [line.strip().split() for line in f.readlines() if line.strip()]
        if filterfile_laval is not None:
            if isinstance(filterfile_laval, str):
                with open(filterfile_laval, 'r') as f:
                    self.filtered_items_laval = [line.strip().split() for line in f.readlines() if line.strip()]
            else:
                assert isinstance(filterfile_laval, list)
                self.filtered_items_laval = []
                for file in filterfile_laval:
                    with open(file, 'r') as f:
                        self.filtered_items_laval.extend([line.strip().split() for line in f.readlines() if line.strip()])
            print("found %d filter items (laval), now filtering" %(len(self.filtered_items_laval)))
            for item in self.items_laval[:]: # NOTE: if not iterating on the new list ([:]), the filtering is incomplete!
                if item in self.filtered_items_laval:
                    self.items_laval.remove(item)

        # merge data
        self.items = []
        for item_syn in self.items_syn:
            self.items.append({"type":"syn", "data":item_syn})
        for item_laval in self.items_laval:
            self.items.append({"type":"laval", "data":item_laval})

        print("dataset phase: %s" %(phase))
        print("num items: %d (synthetic: %d, laval: %d)" %(len(self.items), len(self.items_syn), len(self.items_laval)))

    def get_item(self, idx):
        item = self.items[idx]
        if item['type'] == 'syn':
            city_cam_name, sky_name, angle_id = item['data']
            hdr_path = os.path.join(self.dataroot_syn, 'global_lighting', city_cam_name, 'rendered', sky_name, 'global_lighting_%s.hdr' %(angle_id))
            hdr = imageio.imread(hdr_path, format='HDR-FI') # H * W * C
            hdr = np.transpose(np.asfarray(hdr, dtype=np.float32), (2, 0, 1))*4.0 # C * H * W # NOTE: our envmaps needs a global scale
            Ci, Hi, Wi = hdr.shape
            return_dict = {
                'color': hdr,
                'type': 'synthetic',
                'metadata': ' '.join([city_cam_name, sky_name, angle_id])
            }
            sun_param_path = os.path.join(self.dataroot_syn, 'global_lighting', city_cam_name, 'rendered', sky_name, 'sun_pos_%s.pkl' %(angle_id))
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

        elif item['type'] == 'laval':
            date, time = item['data']
            hdr_path = os.path.join(self.dataroot_laval, date, time, 'envmap.exr')
            hdr = imageio.imread(hdr_path, format='EXR') # H * W * C
            hdr = np.transpose(np.asfarray(hdr, dtype=np.float32), (2, 0, 1)) # C * H * W
            Ci, Hi, Wi = hdr.shape
            hdr = hdr[:,:Hi//2,:]
            Hi = Hi//2
            return_dict = {
                'color': hdr,
                'type': 'laval',
                'metadata': ' '.join([date, time])
            }
            if self.dataroot_sun_laval is not None:
                sun_param_path = os.path.join(self.dataroot_sun_laval, date, time, 'sun_param.pkl')
                with open(sun_param_path, 'rb') as f:
                    sun_params = pickle.load(f)
                if not hasattr(self.opt, 'sunmask_scale'):
                    self.sunmask_scale = 4
                else:
                    self.sunmask_scale = self.opt.sunmask_scale
                pos_mask = np.zeros((1, Hi//self.sunmask_scale, Wi//self.sunmask_scale), dtype=np.float32)
                pos_mask_coarse = np.zeros((1, Hi//(2*self.sunmask_scale), Wi//(2*self.sunmask_scale)), dtype=np.float32)
                pos_mask_fine = np.zeros((1, Hi, Wi), dtype=np.float32)
                if sun_params['visible']: # NOTE: if cloudy, set sunpos mask all zeros!
                    pos_mask[0][int(sun_params['location_xy'][1]//self.sunmask_scale)][int(sun_params['location_xy'][0]//self.sunmask_scale)] = 1.0
                    pos_mask_coarse[0][int(sun_params['location_xy'][1]//(2*self.sunmask_scale))][int(sun_params['location_xy'][0]//(2*self.sunmask_scale))] = 1.0
                    pos_left_ind = max(int(0), int(sun_params['location_xy'][0])-3)
                    pos_right_ind = min(int(128), pos_left_ind+8)
                    pos_left_ind = pos_right_ind - 8
                    pos_upper_ind = max(int(0), int(sun_params['location_xy'][1])-3)
                    pos_lower_ind = min(int(32), pos_upper_ind+8)
                    pos_upper_ind = pos_lower_ind - 8
                    pos_mask_fine[0, pos_upper_ind:pos_lower_ind, pos_left_ind:pos_right_ind] = 1.0
                return_dict['sun_pos_mask'] = pos_mask
                return_dict['sun_pos_mask_coarse'] = pos_mask_coarse
                return_dict['sun_pos_mask_fine'] = pos_mask_fine
                return_dict['sun_vis'] = 0.0 if sun_params['visible'] is False else 1.0
                return_dict['sun_x'] = np.clip((sun_params['location_xy'][0]+0.5) / Wi, 0.0, 1.0)
                return_dict['sun_y'] = np.clip((sun_params['location_xy'][1]+0.5) / Hi, 0.0, 1.0)

        return return_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)
