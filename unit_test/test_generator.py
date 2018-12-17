import matplotlib.pyplot as plt
import numpy as np
import torch
from data import dans_dataset
from data import data_loader
from options.train_options import TrainOptions
from util import util
from models import networks

## make a generator
net_G = networks.define_G(1, 1, 6)
print('made net')
print(len(net_G.models))

## make a dataset
from PIL import Image

test_im = np.array(Image.open('test_ims/aachen_64.png'))
test_im = torch.tensor(test_im)
test_im = test_im.unsqueeze(0)
test_im = test_im.unsqueeze(0)

print('opened image')

# do an actual dataset
opts = TrainOptions().parse()
opts.dataroot = '../datasets/resolutions/'
print(opts.lod_transition_img)
opts.lod_transition_img = 3
opts.lod_train_img = 3

dl = data_loader.CreateDataLoader(opts)
dataset = dl.load_data()

print('made dataloader')


def get_phase(iter, opts):
    transition_img = opts.lod_transition_img
    train_img = opts.lod_train_img
    phase_dur = transition_img + train_img
    phase_idx = iter//phase_dur
    phase_img = iter - phase_idx*phase_dur

    phase = phase_idx + 1
    if phase_img < train_img:
        alpha = 1
    else:
        alpha = 1 - (phase_img - train_img)/transition_img

    return phase, alpha


def test_getPhase():
    print('test getPhase')
    for i in range(20):
        print(get_phase(i, opts))
    print()


phase_params = []
for iter, data in enumerate(dataset):
    test_im = data['s64']
    t_size = test_im.size()
    #print(t_size)
    ## pass some images into the generator
    phase, alpha = get_phase(iter, opts)
    print(phase, alpha)
    phase_params.append((phase, alpha))
    # for i in net_G.state_dict():
    #    print(i)
    fake = net_G.forward(test_im, phase, alpha)
    print('generated fake, with dims: ', fake.size())
    if iter == 3:
        print('testing at max phase')
        fake = net_G.forward(test_im, 6, 0)
        break

fake = fake.detach()[0, ...]
print(fake.size())
## view them
im = util.tensor2im(fake)
plt.imshow(im)
plt.show()
