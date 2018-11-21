import numpy as np
import util as u
import matplotlib.pyplot as plt


def check_labels(im,show=True):
    try:
        im = u.util.tensor2im(im, normalize=False)
    except AttributeError as e:
        print(e)
        print('not a tensor - ', type(im))

    n_zeros = im[im < 1].size
    n_elems = im.size

    if n_zeros == n_elems:
        if show:
            do_show(im)
        raise ValueError('image all 0s!')
    else:
        if show:
            do_show(im)
        print('image good! size {}, zeros {}'.format(n_elems,n_zeros))


def do_show(im):
    print(im.shape)
    plt.imshow(im)
    plt.show()

def test_label(f):
    from PIL import Image
    im = Image.open(f)
    im = np.array(im)
    check_labels(im)


#test_label('../checkpoints/block2label/web/images/epoch002_real_image.jpg')
