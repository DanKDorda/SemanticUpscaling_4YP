import matplotlib.pyplot as plt
import numpy as np
import torch
from data import data_loader
from options.train_options import TrainOptions
from util import util
from models import networks

# make a discriminator net
netD = networks.define_D(1, 64, 3)
print('made net')

# get some data to feed into it
opts = TrainOptions().parse()
opts.dataroot = '../datasets/resolutions/'
print(opts.lod_transition_img)
opts.lod_transition_img = 3
opts.lod_train_img = 3

dl = data_loader.CreateDataLoader(opts)
dataset = dl.load_data()
print('made dataloader')

# test the discriminator
done_checks = False

for iter, data in enumerate(dataset):
    test_im = data['s64']
    # netD outputs a list of lists of tensors
    result = netD(test_im)

    if not done_checks:
        print('input im size: ', test_im.size())
        print()
        print(type(test_im))
        print(type(result))
        print(len(result))
        print(type(result[0]))
        print(type(result[0][0]))
        real_result = result[0][0]
        print('output size: ', real_result.size())


    done_checks = True
    if iter == 3:
        break

print('\ndone')
# observe the output
