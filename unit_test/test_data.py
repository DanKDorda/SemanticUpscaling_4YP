import matplotlib.pyplot as plt
from data import data_loader
from options.train_options import TrainOptions
from util import util

opts = TrainOptions().parse()
opts.dataroot = '../datasets/resolutions/'

dl = data_loader.CreateDataLoader(opts)
dataset = dl.load_data()
print('made dataloader')


def print_data_subset():
    for i, b in enumerate(dataset):
        if i > 4:
            break
        # read some vals
        print('im looopin')

        s1 = b['s1']
        s16 = b['s16']
        s64 = b['s64']

        s1 = util.tensor2im(s1[0, :, :, :], normalize=False)
        s16 = util.tensor2im(s16[0, :, :, :])
        s64 = util.tensor2im(s64[0, :, :, :])
        # plt them
        plt.subplot(1, 3, 1)
        plt.imshow(s1)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(s16)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(s64)
        plt.axis('off')

        plt.show()

    print('done')


def investigate_uniques():
    import torch
    uniques = torch.Tensor()
    prev_size = len(uniques)
    for i, dat in enumerate(dataset):
        if i > 100:
            break

        s4 = dat['s4']

        current_uniques = torch.unique(s4)
        uniques = torch.unique(torch.cat((uniques, current_uniques), 0), sorted=True)
        if i % 10 == 0:
            print('updated with {} uniques vals'.format(len(uniques) - prev_size))
            prev_size = len(uniques)

    print(uniques)
    prev_size = len(uniques)
    print('num uniques: ', prev_size)


investigate_uniques()
