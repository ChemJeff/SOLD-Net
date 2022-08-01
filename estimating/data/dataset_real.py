import os
import numpy as np
import imageio
import glob
from torch.utils.data import Dataset

class RealDataset(Dataset):
    def __init__(self, opt, dataroot):
        self.opt = opt
        self.dataroot = dataroot
        _tmp_list = [os.path.split(path)[-1] for path in sorted(glob.glob(os.path.join(dataroot, '*_persp.*')))]
        self.items = []
        for persp in _tmp_list:
            if os.path.exists(os.path.join(dataroot, persp.lower().replace('_persp.jpg', '_local.npy'))) and os.path.exists(os.path.join(dataroot, persp.lower().replace('_persp.jpg', '_local_0.hdr'))) and os.path.exists(os.path.join(dataroot, persp.lower().replace('_persp.jpg', '_local_1.hdr'))):
                self.items.append(persp.split('_')[0])
        print("dataset phase: real data test")
        print("num items: %d" %(len(self.items)))

    def get_item(self, idx):
        item = self.items[idx]
        img_idx = item
        persp_ldr_path = os.path.join(self.dataroot, '%s_persp.JPG' %(img_idx))
        persp = imageio.imread(persp_ldr_path, format='JPG') # H * W * C
        persp = np.transpose(np.asfarray(persp, dtype=np.float32), (2, 0, 1)) / 255.0 # C * H * W
        local_pos_path = os.path.join(self.dataroot, '%s_local.npy' %(img_idx))
        local_pos = np.load(local_pos_path)
        local_panos = []
        for local_id in range(local_pos.shape[0]):
            _local_pano_path = os.path.join(self.dataroot, '%s_local_%d.hdr' %(img_idx, local_id))
            _local_pano = imageio.imread(_local_pano_path, format='HDR-FI') # H * W * C
            _local_pano = np.transpose(np.asfarray(_local_pano, dtype=np.float32), (2, 0, 1)) # C * H * W
            local_panos.append(_local_pano)
        local_panos = np.stack(local_panos, axis=0)
        if os.path.exists(os.path.join(self.dataroot, '%s_sun_pos.txt' %(img_idx))):
            sun_pos = np.loadtxt(os.path.join(self.dataroot, '%s_sun_pos.txt' %(img_idx)))
        else:
            sun_pos = np.array([-1, -1])
        return_dict = {
            'color': persp,
            'local_pos': local_pos,
            'local_pano': local_panos,
            'sun_pos': sun_pos,
            'is_sunny': 1.0 if sun_pos[0] >= 0 else 0.0,
            'meta': img_idx
        }
        return return_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)
