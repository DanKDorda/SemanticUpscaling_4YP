import matplotlib.pyplot as plt
import numpy as np
import torch
from data import data_loader
from options.train_options import TrainOptions
from util import util
from models import networks

# make a generator net

netD = networks.define_D(1, 64, 3)
print('made net')


# get some data to feed into it
data = []

result = netD(data)
print(result)

# observe the output