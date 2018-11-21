import numpy as np
import torch
import torch.nn as nn


###############################################################################
# Functions
###############################################################################

def define_G():
    pass


def define_D():
    pass


def print_network(net):
    pass


################################################################################
# Losses
################################################################################
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()


class LabelLoss():
    def __init__(self):
        pass


################################################################################
# Generators
################################################################################
class GlobalGenerator(nn.Module):
    def __init__(self):
        super(GlobalGenerator, self).__init__()
