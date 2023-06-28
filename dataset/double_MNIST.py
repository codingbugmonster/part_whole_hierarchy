import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import os.path
np.random.seed(9825619)

data_dir = "../tmp_data"

# Load the MNIST Dataset as prepared by the brainstorm data script
# You will need to run brainstorm/data/create_mnist.py first
with h5py.File(os.path.join(data_dir, 'MNIST.hdf5'), 'r') as f:
    mnist_digits = f['normalized_full/training/default'][0, :]
    mnist_targets = f['normalized_full/training/targets'][:]
    mnist_digits_test = f['normalized_full/test/default'][0, :]
    mnist_targets_test = f['normalized_full/test/targets'][:]

def crop(d):
    return d[np.sum(d, 1) != 0][:, np.sum(d, 0) != 0]

def make_double(digit1,digit2, binarize_threshold=0.5):
    sy1, sx1 = digit1.shape
    sy2, sx2 = digit2.shape
    # print(sy1,sx1,sy2,sx2)
    w = sx1+sx2+4
    h = max(sy1,sy2)+4
    img = np.zeros((h,w))
    grp_d = np.zeros((h, w))
    x1 = np.random.randint(0, 2)
    y1 = np.random.randint(0, 4)
    x2 = np.random.randint(x1+sx1, x1+sx1+2)
    y2 = np.random.randint(0, 4)
    region1 = (slice(y1, y1 + sy1), slice(x1, x1 + sx1))
    region2 = (slice(y2, y2 + sy2), slice(x2, x2 + sx2))
    m1 = digit1 >= binarize_threshold
    # print(y1, y1 + sy1,x1, x1 + sx1)
    # print(y2, y2 + sy2, x2, x2 + sx2)
    # print(img.shape, img[region1].shape)
    img[region1][m1] = 1
    grp_d[region1][m1] = 1
    # print(img.shape,img[region1].shape)
    m2 = digit2 >= binarize_threshold
    # print(img.shape, img[region2].shape)
    # print(y2+sy2,x2+sx2,w,h)
    img[region2][m2] = 1
    grp_d[region2][m2] = 2
    return img, grp_d

def generate_multi_double_mnist_img(digit_nrs1,digit_nrs2, size=(60, 60), test=False, binarize_threshold=0.5):
    if not test:
        digits1 = [crop(mnist_digits[nr].reshape(28, 28)) for nr in digit_nrs1]
        digits2 = [crop(mnist_digits[nr].reshape(28, 28)) for nr in digit_nrs2]
    else:
        digits1 = [crop(mnist_digits_test[nr].reshape(28, 28)) for nr in digit_nrs1]
        digits2 = [crop(mnist_digits_test[nr].reshape(28, 28)) for nr in digit_nrs2]

    flag = False
    while not flag:
        img = np.zeros(size)
        grp = np.zeros(size)
        grp_p = np.zeros(size)
        mask = np.zeros(size)
        k = 1
        l=1

        for i in range(len(digits1)):
            h, w = size
            double, grp_d = make_double(digits1[i],digits2[i])
            sy, sx = double.shape
            x = np.random.randint(0, w - sx + 1)
            y = np.random.randint(0, h - sy + 1)
            region = (slice(y, y + sy), slice(x, x + sx))
            m = double >= binarize_threshold
            m1 = grp_d == 1
            m2 = grp_d == 2
            img[region][m] = 1
            mask[region][m] += 1
            grp[region][m] = k
            grp_p[region][m1] = l
            l+=1
            grp_p[region][m2] = l
            l+=1
            k += 1
        if len(digit_nrs1) <= 1 or (mask[region][m] > 1).sum() / (mask[region][m] >= 1).sum() < 0.2:
            flag = True

    grp[mask > 1] = 0  # ignore overlap regions
    grp_p[mask > 1] = 0  # ignore overlap regions
    return img, grp, grp_p


def generate_multi_mnist_img(digit_nrs, size=(60, 60), test=False, binarize_threshold=0.5):
    if not test:
        digits = [crop(mnist_digits[nr].reshape(28, 28)) for nr in digit_nrs]
    else:
        digits = [crop(mnist_digits_test[nr].reshape(28, 28)) for nr in digit_nrs]

    flag = False
    while not flag:
        img = np.zeros(size)
        grp = np.zeros(size)
        mask = np.zeros(size)
        k = 1

        for i, digit in enumerate(digits):
            h, w = size
            sy, sx = digit.shape
            x = np.random.randint(0, w - sx + 1)
            y = np.random.randint(0, h - sy + 1)
            region = (slice(y, y + sy), slice(x, x + sx))
            m = digit >= binarize_threshold
            img[region][m] = 1
            mask[region][m] += 1
            grp[region][m] = k
            k += 1
        if len(digit_nrs) <= 1 or (mask[region][m] > 1).sum() / (mask[region][m] >= 1).sum() < 0.2:
            flag = True

    grp[mask > 1] = 0  # ignore overlap regions
    return img, grp, grp


fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(16, 5))
for ax in axes.T:
    digit_nrs = np.random.randint(0, 60000, 2)
    img, grp, grp2 = generate_multi_mnist_img(digit_nrs,size=(80,80))
    plot_input_image(img, ax[0])
    plot_groups(grp, ax[1])
    plot_groups(grp2, ax[2])
plt.savefig("../tmp_img/mnist_part.png")

fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(16, 5))
for ax in axes.T:
    digit_nrs1 = np.random.randint(0, 60000, 2)
    digit_nrs2 = np.random.randint(0, 60000, 2)
    img, grp, grp2 = generate_multi_double_mnist_img(digit_nrs1,digit_nrs2,size=(80,80))
    plot_input_image(img, ax[0])
    plot_groups(grp, ax[1])
    plot_groups(grp2, ax[2])
plt.savefig("../tmp_img/mnist_whole.png")


np.random.seed(36520)
nr_digits = 2
mnist_size = 60000
nr_training_examples = 60000
nr_test_examples = 10000
nr_single_examples = 60000
size = (80, 80)

data = np.zeros((60000,) + size, dtype=np.float32)
grps = np.zeros_like(data)
grps2 = np.zeros_like(data)

ddata = np.zeros((60000,) + size, dtype=np.float32)
dgrps = np.zeros_like(data)
dgrps2 = np.zeros_like(data)

# targets = np.zeros((60000, nr_digits), dtype=np.int)
for i in range(60000):
    digit_nrs = np.random.randint(0, 60000, nr_digits)
    data[i, :, :], grps[i, :, :], grps2[i, :, :] = generate_multi_mnist_img(digit_nrs, size=size)

    igit_nrs1 = np.random.randint(0, 60000, nr_digits)
    igit_nrs2 = np.random.randint(0, 60000, nr_digits)
    ddata[i, :, :], dgrps[i, :, :], dgrps2[i, :, :] = generate_multi_double_mnist_img(igit_nrs1,igit_nrs2, size=size)
    # targets[i, :] = mnist_targets[0, digit_nrs, 0]

data_test = np.zeros((10000,) + size, dtype=np.float32)
grps_test = np.zeros_like(data_test)
grps2_test = np.zeros_like(data_test)
ddata_test = np.zeros((10000,) + size, dtype=np.float32)
dgrps_test = np.zeros_like(data_test)
dgrps2_test = np.zeros_like(data_test)
# targets_test = np.zeros((1, 10000, nr_digits), dtype=np.int)
for i in range(10000):
    digit_nrs = np.random.randint(0, 10000, nr_digits)
    data_test[i, :, :], grps_test[i, :, :], grps2_test[i, :, :] = generate_multi_mnist_img(digit_nrs, size=size, test=True)
    # targets_test[0, i, :] = mnist_targets_test[0, digit_nrs, 0]
    digit_nrs1 = np.random.randint(0, 10000, nr_digits)
    digit_nrs2 = np.random.randint(0, 10000, nr_digits)
    ddata_test[i, :, :], dgrps_test[i, :, :], dgrps2_test[i, :, :] = generate_multi_double_mnist_img(digit_nrs1,digit_nrs2, size=size, test=True)

data_single = np.zeros((nr_single_examples,) + size, dtype=np.float32)
grps_single = np.zeros_like(data_single )
grps2_single = np.zeros_like(data_single )
# targets_single = np.zeros((1, nr_single_examples, 1), dtype=np.int)
ddata_single = np.zeros((nr_single_examples,) + size, dtype=np.float32)
dgrps_single = np.zeros_like(data_single )
dgrps2_single = np.zeros_like(data_single )
for i in range(nr_single_examples):
    data_single [i, :, :], grps_single[i, :, :], grps2_single[i, :, :] = generate_multi_mnist_img([i % mnist_size], size=size)
    # targets_single[0, i, :] = mnist_targets[0, i, 0]
    digit_nrs1 = np.random.randint(0, 10000)
    digit_nrs2 = np.random.randint(0, 10000)
    ddata_single[i, :, :], dgrps_single[i, :, :], dgrps2_single[i, :, :] = generate_multi_double_mnist_img([digit_nrs1],[digit_nrs2], size=size)

with h5py.File(os.path.join(data_dir, 'mnist_part.h5'), 'w') as f:
    single = f.create_group('train_single')
    single.create_dataset('default', data=data_single, compression='gzip', chunks=(100,) + size)
    # single.create_dataset('groups', data=grps_single, compression='gzip', chunks=(100,) + size)
    single.create_dataset('groups1', data=grps_single, compression='gzip', chunks=(100,) + size)
    single.create_dataset('groups2', data=grps2_single, compression='gzip', chunks=(100,) + size)
    # single.create_dataset('targets', data=targets_single, compression='gzip', chunks=(1, 100, 1))

    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(100,) + size)
    # train.create_dataset('groups', data=grps, compression='gzip', chunks=(100,) + size)
    train.create_dataset('groups1', data=grps, compression='gzip', chunks=(100,) + size)
    train.create_dataset('groups2', data=grps2, compression='gzip', chunks=(100,) + size)
    # train.create_dataset('targets', data=targets, compression='gzip', chunks=(1, 100, nr_digits))

    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(100,) + size)
    # test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(100,) + size)
    test.create_dataset('groups1', data=grps_test, compression='gzip', chunks=(100,) + size)
    test.create_dataset('groups2', data=grps2_test, compression='gzip', chunks=(100,) + size)
    # test.create_dataset('targets', data=targets_test, compression='gzip', chunks=(1, 100, nr_digits))

with h5py.File(os.path.join(data_dir, 'mnist_whole.h5'), 'w') as f:
    single = f.create_group('train_single')
    single.create_dataset('default', data=ddata_single, compression='gzip', chunks=(100,) + size)
    # single.create_dataset('groups', data=dgrps_single, compression='gzip', chunks=(100,) + size)
    single.create_dataset('groups1', data=dgrps_single, compression='gzip', chunks=(100,) + size)
    single.create_dataset('groups2', data=dgrps2_single, compression='gzip', chunks=(100,) + size)
    # single.create_dataset('targets', data=targets_single, compression='gzip', chunks=(1, 100, 1))

    train = f.create_group('train_multi')
    train.create_dataset('default', data=ddata, compression='gzip', chunks=(100,) + size)
    # train.create_dataset('groups', data=dgrps, compression='gzip', chunks=(100,) + size)
    train.create_dataset('groups1', data=dgrps, compression='gzip', chunks=(100,) + size)
    train.create_dataset('groups2', data=dgrps2, compression='gzip', chunks=(100,) + size)
    # train.create_dataset('targets', data=targets, compression='gzip', chunks=(1, 100, nr_digits))

    test = f.create_group('test')
    test.create_dataset('default', data=ddata_test, compression='gzip', chunks=(100,) + size)
    # test.create_dataset('groups', data=dgrps_test, compression='gzip', chunks=(100,) + size)
    test.create_dataset('groups1', data=dgrps_test, compression='gzip', chunks=(100,) + size)
    test.create_dataset('groups2', data=dgrps2_test, compression='gzip', chunks=(100,) + size)
    # test.create_dataset('targets', data=targets_test, compression='gzip', chunks=(1, 100, nr_digits))
