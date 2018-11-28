### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image


class DansDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        dir_1 = 's1'
        dir_2 = 's2'
        dir_4 = 's4'
        dir_8 = 's8'
        dir_16 = 's16'
        dir_32 = 's32'
        dir_64 = 's64'

        self.dir_1 = os.path.join(opt.dataroot, 's1')
        self.dir_2 = os.path.join(opt.dataroot, 's2')
        self.dir_4 = os.path.join(opt.dataroot, 's4')
        self.dir_8 = os.path.join(opt.dataroot, 's8')
        self.dir_16 = os.path.join(opt.dataroot, 's16')
        self.dir_32 = os.path.join(opt.dataroot, 's32')
        self.dir_64 = os.path.join(opt.dataroot, 's64')

        self.paths1 = sorted(make_dataset(self.dir_1))
        self.paths2 = sorted(make_dataset(self.dir_2))
        self.paths4 = sorted(make_dataset(self.dir_4))
        self.paths8 = sorted(make_dataset(self.dir_8))
        self.paths16 = sorted(make_dataset(self.dir_32))
        self.paths32 = sorted(make_dataset(self.dir_16))
        self.paths64 = sorted(make_dataset(self.dir_64))

        self.dataset_size = len(self.paths1)

    def __getitem__(self, index):
        ### input A (label maps)
        s1 = Image.open(self.paths1[index])
        s2 = Image.open(self.paths2[index])
        s4 = Image.open(self.paths4[index])
        s8 = Image.open(self.paths8[index])
        s16 = Image.open(self.paths16[index])
        s32 = Image.open(self.paths32[index])
        s64 = Image.open(self.paths64[index])

        params = get_params(self.opt, s1.size)

        def get_tensor(A):
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            return transform_A(A) * 255.0

        sd = {'s1': s1, 's2': s2, 's4': s4, 's8': s8, 's16': s16, 's32': s32, 's64': s64}
        trans_dict = {}

        for k, v in sd.items():
            trans_dict[k] = get_tensor(v)

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)

        trans_dict['paths'] = 'umm not sure'
        trans_dict['inst'] = 0
        trans_dict['feat'] = 0
        return trans_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'DansAlignedDataset'
