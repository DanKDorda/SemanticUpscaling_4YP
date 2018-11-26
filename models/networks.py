import numpy as np
import torch
import torch.nn as nn


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_G(input_nc, output_nc, n_phases):
    netG = GlobalGenerator(input_nc, output_nc, n_phases)
    netG.apply(weights_init)

    return netG


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
    def __init__(self, input_nc, output_nc, n_phases, ngf=64, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_phases > 0)
        super(GlobalGenerator, self).__init__()
        self.n_phases = n_phases
        self.models = []

        activation = nn.ReLU(True)

        ## add first layer
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.models += nn.Sequential(*model)
        ## add layers for phases
        for n in range(n_phases - 2):
            mult = 1
            # mult = 2**n
            # down
            model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                     norm_layer(ngf * mult * 2), activation]
            # res
            model += [ResnetBlock(ngf, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            # up
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
            self.models += nn.Sequential(*model)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1), norm_layer(ngf), activation]
        self.models += nn.Sequential(*model)

    def forward(self, input, phase, blending):
        assert phase <= self.n_phases

        output = self.models[0](input)
        for idx, model in enumerate(self.models[1:phase]):
            if idx == phase:
                output = (1 - blending) * output + blending * model(output)
            else:
                output = model(output)

        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
