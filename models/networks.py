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
        self.final_layers = []

        def get_model_block(ngf, mult, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), padding_type='reflect'):
            # down
            model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=2),
                     norm_layer(ngf * mult * 2), activation]
            # res
            model += [ResnetBlock(ngf * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            # up
            model += [nn.ConvTranspose2d(ngf * mult * 2, ngf * mult, kernel_size=4, stride=2, padding=2),
                      norm_layer(ngf * mult), activation]
            return model

        activation = nn.ReLU(True)

        ## add first layer 1 -> 64
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1), norm_layer(ngf), activation]
        out_layer = nn.Sequential(nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1))
        self.models += [(nn.Sequential(*model), out_layer)]
        ## add layers for phases

        for n in range(n_phases - 1):
            mult = 1
            # mult = 2**n
            model = get_model_block(ngf, mult, norm_layer, activation, padding_type)
            # final 64 -> 1 layer
            self.models += [(nn.Sequential(*model), out_layer)]

    def forward(self, input, end_phase, blend_prev):
        assert end_phase <= self.n_phases, 'phase is: {}, max: {}'.format(end_phase, self.n_phases)

        # 1 -> 64
        output = self.models[0][0](input)
        # 64 -> 64, until time comes for out, then 64 -> 64 -> 1 with blend of previous ->1
        for n in range(1, end_phase):
            print(n)
            model = self.models[n]
            if n == end_phase - 1:
                print('out at phase: ', n)
                output = blend_prev * self.models[n - 1][1](output) + (1 - blend_prev) * model[1](model[0](output))
            else:
                output = model[0](output)

        # for idx, model in enumerate(self.models[1:end_phase]):
        #     if idx == end_phase:
        #         output = (1 - blending) * output + blending * model(output)
        #         break
        #     else:
        #         output = model(output)

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
