import matplotlib.pyplot as plt
from data import aligned_dataset
from data import data_loader
from options.train_options import TrainOptions
from util import util

opts = TrainOptions().parse()
opts.dataroot = '../datasets/resolutions/'

dl = data_loader.CreateDataLoader(opts)
dataset = dl.load_data()
print('made dataloader')

for i, b in enumerate(dataset):
    if i > 4:
        break
    # read some vals
    print('im looopin')

    s1 = b['s1_tensor']
    s16 = b['s16_tensor']
    s64 = b['s64_tensor']

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