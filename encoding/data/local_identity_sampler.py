import os
import copy
import random
import numpy as np

from collections import defaultdict
from torch.utils.data import Sampler

class RandomLocalIdentiyiSampler(Sampler):
    '''
    modified for LSDataser class
    '''
    def __init__(self, data_source, dataroot, batch_size, num_instance ):
        self.data_source = data_source
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.num_pids_per_batch = self.batch_size // self.num_instance
        self.index_dic = defaultdict(list)
 
        self.local_env_splits = []
        self.env_to_split = {}
        for i in range(4):
            local_summary_file = os.path.join(dataroot, 'local_cameras', '.summary', 'local_cameras_%d_of_4.txt' %(i))
            with open(local_summary_file, 'r') as f:
                _envs = f.readline().strip().split()
            self.local_env_splits.append(_envs)
            for _env in _envs:
                self.env_to_split[_env] = i

        for index, item in enumerate(self.data_source.items):
            city_cam_name, sky_name, angle_id, local_cam_id = item
            env_split = self.env_to_split[sky_name]
            pid = "".join([city_cam_name, str(env_split), local_cam_id])
            self.index_dic[pid].append(index)
        
        self.pids = list(self.index_dic.keys())
 
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instance:
                num = self.num_instance
            self.length += num - num % self.num_instance # drop last, pad shortage
 
 
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
 
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instance:
                idxs = np.random.choice(idxs, size = self.num_instance, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if(len(batch_idxs) == self.num_instance):
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
 
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while(len(avai_pids)>=self.num_pids_per_batch):
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid])==0:
                    avai_pids.remove(pid)
 
        self.length = len(final_idxs)
        return iter(final_idxs)
 
    def __len__(self):
        return self.length
