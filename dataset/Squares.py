"""
Corner problem:

Reference
[1] David P. Reichert and Thomas Serre, Neuronal Synchrony in Complex-Valued Deep Networks, ICLR 2014
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import os.path

np.random.seed(746519283)

width = 28
height = 28

corner = np.zeros((5, 5))
corner[:2, :] = 1.0
corner[:, :2] = 1.0

corners = [
    corner.copy(),
    corner[::-1, :].copy(),
    corner[:, ::-1].copy(),
    corner[::-1, ::-1].copy()
]

square = np.zeros((15, 15))
square[:5, :5] = corners[0]
square[-5:, :5] = corners[1]
square[:5, -5:] = corners[2]
square[-5:, -5:] = corners[3]

def generate_corners_image(width, height, nr_squares=1, nr_corners=4):
    img = np.zeros((height, width))
    grp = np.zeros_like(img)
    grp_p = np.zeros_like(img)
    k = 1
    l = 1
    
    for i in range(nr_squares):
        x = np.random.randint(0, width-19)
        y = np.random.randint(0, height-19)
        region = (slice(y,y+15), slice(x,x+15))
        img[region][square != 0] += 1
        grp[region][square != 0] = k        
        k += 1
        grp_p[region][:5, :5][square[:5, :5] != 0] = l
        l+=1
        grp_p[region][-5:, :5][square[-5:, :5] != 0] = l
        l+=1
        grp_p[region][:5, -5:][square[:5, -5:] != 0] = l
        l+=1
        grp_p[region][-5:, -5:][square[-5:, -5:] != 0] = l
        l+=1



    for i in range(nr_corners):
        x = np.random.randint(0, width-4)
        y = np.random.randint(0, height-4)
        corner = corners[np.random.randint(0, 4)]
        region = (slice(y,y+5), slice(x,x+5))
        img[region][corner != 0] += 1
        grp[region][corner != 0] = k
        grp_p[region][corner != 0] = k
        k += 1
        
    grp[img > 1] = 0
    grp_p[img > 1] = 0
    img = img != 0
    return img, grp, grp_p

fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(16, 5))
for ax in axes.T:
    img1, grp1, grp12 = generate_corners_image(60, 60, 1, 0)
    plot_input_image(img1, ax[0])
    plot_groups(grp1, ax[1])
    plot_groups(grp12, ax[2])
plt.savefig("../tmp_img/C_square.png")

fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(16, 5))
for ax in axes.T:
    img2, grp2, grp22 = generate_corners_image(60, 60, 3, 0)
    plot_input_image(img2, ax[0])
    plot_groups(grp2, ax[1])
    plot_groups(grp22, ax[2])
plt.savefig("../tmp_img/C_squares.png")

fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(16, 5))
for ax in axes.T:
    img3, grp3, grp32 = generate_corners_image(60, 60, 0, 1)
    plot_input_image(img3, ax[0])
    plot_groups(grp3, ax[1])
    plot_groups(grp32, ax[2])
plt.savefig("../tmp_img/C_corners.png")

data_dir = "../tmp_data"

nr_train_examples = 60000
nr_test_examples = 10000
nr_single_examples = 60000

width = 60
height = 60
nr_squares = 3
nr_corners = 0

data = np.zeros((nr_train_examples, height, width), dtype=np.float32)
# grps = np.zeros_like(data)
grps1 = np.zeros_like(data)
grps2 = np.zeros_like(data)

for i in range(nr_train_examples):
    # data[i, :, :], grps[i, :, :] = generate_corners_image(width, height, nr_squares, nr_corners)
    data[i, :, :], grps1[i, :, :], grps2[i, :, :] = generate_corners_image(width, height, nr_squares, nr_corners)
    
data_test = np.zeros((nr_test_examples, height, width), dtype=np.float32)
# grps_test = np.zeros_like(data_test)
grps1_test = np.zeros_like(data_test)
grps2_test = np.zeros_like(data_test)

for i in range(nr_test_examples):
    # data_test[i, :, :], grps_test[i, :, :] = generate_corners_image(width, height, nr_squares,
    #                                                                             nr_corners)
    data_test[i, :, :], grps1_test[i, :, :], grps2_test[i, :, :] = generate_corners_image(width, height, nr_squares,
                                                                    nr_corners)

data_single1 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
# grps_single1 = np.zeros_like(data_single1)
grps1_single1 = np.zeros_like(data_single1)
grps2_single1 = np.zeros_like(data_single1)

for i in range(nr_single_examples):
    # data_single1[i, :, :], grps_single1[i, :, :] = generate_corners_image(width, height, 0, 1)
    data_single1[i, :, :], grps1_single1[i, :, :], grps2_single1[i, :, :] = generate_corners_image(width, height, 0, 1)

data_single2 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
# grps_single2 = np.zeros_like(data_single1)
grps1_single2 = np.zeros_like(data_single1)
grps2_single2 = np.zeros_like(data_single1)
for i in range(nr_single_examples):
    # data_single2[i, :, :], grps_single2[i, :, :] = generate_corners_image(width, height, 1, 0)
    data_single2[i, :, :], grps1_single2[i, :, :], grps2_single2[i, :, :] = generate_corners_image(width, height, 1, 0)


# shuffel_idx = np.arange(nr_single_examples)
# np.random.shuffle(shuffel_idx)
# data_single = data_single[shuffel_idx, :]
# grps_single = grps_single[shuffel_idx, :]

with h5py.File(os.path.join(data_dir, 'corners_part.h5'), 'w') as f:
    single1 = f.create_group('train_single')
    single1.create_dataset('default', data=data_single1, compression='gzip', chunks=(100, height, width))
    # single1.create_dataset('groups', data=grps_single1, compression='gzip', chunks=(100, height, width))
    single1.create_dataset('groups1', data=grps1_single1, compression='gzip', chunks=(100, height, width))
    single1.create_dataset('groups2', data=grps2_single1, compression='gzip', chunks=(100, height, width))

    # single2 = f.create_group('train_single2')
    # single2.create_dataset('default', data=data_single2, compression='gzip', chunks=(100, height, width))
    # single2.create_dataset('groups', data=grps_single2, compression='gzip', chunks=(100, height, width))
    
    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(100, height, width))
    # train.create_dataset('groups', data=grps, compression='gzip', chunks=(100, height, width))
    train.create_dataset('groups1', data=grps1, compression='gzip', chunks=(100, height, width))
    train.create_dataset('groups2', data=grps2, compression='gzip', chunks=(100, height, width))
    
    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(100, height, width))
    # test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(100, height, width))
    test.create_dataset('groups1', data=grps1_test, compression='gzip', chunks=(100, height, width))
    test.create_dataset('groups2', data=grps2_test, compression='gzip', chunks=(100, height, width))


with h5py.File(os.path.join(data_dir, 'corners_whole.h5'), 'w') as f:
    # single1 = f.create_group('train_single1')
    # single1.create_dataset('default', data=data_single1, compression='gzip', chunks=(100, height, width))
    # single1.create_dataset('groups', data=grps_single1, compression='gzip', chunks=(100, height, width))

    single2 = f.create_group('train_single')
    single2.create_dataset('default', data=data_single2, compression='gzip', chunks=(100, height, width))
    # single2.create_dataset('groups', data=grps_single2, compression='gzip', chunks=(100, height, width))
    single2.create_dataset('groups1', data=grps1_single2, compression='gzip', chunks=(100, height, width))
    single2.create_dataset('groups2', data=grps2_single2, compression='gzip', chunks=(100, height, width))

    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(100, height, width))
    # train.create_dataset('groups', data=grps, compression='gzip', chunks=(100, height, width))
    train.create_dataset('groups1', data=grps1, compression='gzip', chunks=(100, height, width))
    train.create_dataset('groups2', data=grps2, compression='gzip', chunks=(100, height, width))

    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(100, height, width))
    # test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(100, height, width))
    test.create_dataset('groups1', data=grps1_test, compression='gzip', chunks=(100, height, width))
    test.create_dataset('groups2', data=grps2_test, compression='gzip', chunks=(100, height, width))