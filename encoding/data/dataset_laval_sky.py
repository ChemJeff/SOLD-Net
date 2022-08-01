import os
import numpy as np
import imageio
import pickle
from torch.utils.data import Dataset

class LavalSkyDataset(Dataset):
    def __init__(self, opt, dataroot, splitfile, phase="train", filterfile=None, dataroot_sun=None):
        assert phase.lower() in ["train", "val", "test"]
        self.opt = opt
        self.phase = phase
        self.dataroot = dataroot
        self.dataroot_sun = dataroot_sun
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
        print("dataset phase: %s" %(phase))
        print("num items: %d" %(len(self.items)))

    def get_item(self, idx):
        item = self.items[idx]
        date, time = item
        hdr_path = os.path.join(self.dataroot, date, time, 'envmap.exr')
        hdr = imageio.imread(hdr_path, format='EXR') # H * W * C
        hdr = np.transpose(np.asfarray(hdr, dtype=np.float32), (2, 0, 1)) # C * H * W
        Ci, Hi, Wi = hdr.shape
        hdr = hdr[:,:Hi//2,:]
        Hi = Hi//2
        return_dict = {
            'color': hdr,
            'color_path': hdr_path,
            'date': date,
            'time': time
        }
        if self.dataroot_sun is not None:
            sun_param_path = os.path.join(self.dataroot_sun, date, time, 'sun_param.pkl')
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
            return_dict['sun_int'] = sun_params['intensity']
            return_dict['sun_size'] = sun_params['size']
        return return_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)