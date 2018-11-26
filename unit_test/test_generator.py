import matplotlib.pyplot as plt
import numpy as np
import torch
from data import aligned_dataset
from data import data_loader
from options.train_options import TrainOptions
from util import util
from models import networks


## make a generator
net_G = networks.define_G(1, 1, 6)
print('made net')

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

dl = data_loader.CreateDataLoader(opts)
dataset = dl.load_data()
print('made dataloader')


## pass some images into the generator
fake = net_G.forward(test_im, 6, 1)
print('generated fake')

## view them
plt.imshow(util.tensor2im(fake))
plt.show()