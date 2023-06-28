"""
Adapted bars problem:

Binary images with a fixed number of randomly placed horizontal and
vertical bars. 
width = height = 20  # 图片大小
nr_norizontal_bars = nr_vertical_bars = 6  # 图片中水平和竖直bar的数量
====================================================================
This python file should run in the diectory where the file exists.
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import copy


def generate_bars(width, height, nr_horizontal_bars, nr_vertical_bars):
    """
    generate test bars
    input:
        width: image width
        height: image height
        nr_horizontal_bars: numbers of horizontal bars
        nv_vertical_bars: numbers of vertical bars
    output:
        img: result image
        grp: mask matrix with 1 been selected; 0, not selected
    """
    img = np.zeros((height, width), dtype=np.float)
    grp = np.zeros_like(img)
    
    idx_vert = np.random.choice(np.arange(width), replace=False, size=nr_vertical_bars)
    img[:, idx_vert] = 1.
    k = 1
    for i in idx_vert:
        grp[:, i] = k
        k += 1
    
    idx_horiz = np.random.choice(np.arange(height), replace=False, size=nr_horizontal_bars)
    img[idx_horiz, :] += 1.
    for i in idx_horiz:
        grp[i, :] = k
        k += 1
    
    grp[img > 1] = 0  # 交叉处颜色为0
    img = img != 0
    
    return img, grp


def generate_bar_segments(width, height, lgh, nr_horizontal_bars, nr_vertical_bars):
    """
    generate test bars
    input:
        width: image width
        height: image height
        nr_horizontal_bars: numbers of horizontal bars
        nv_vertical_bars: numbers of vertical bars
    output:
        img: result image
        grp: mask matrix with 1 been selected; 0, not selected
    """
    img = np.zeros((height, width), dtype=np.float)
    grp = np.zeros_like(img)

    idx_vert = np.random.choice(np.arange(width), replace=False, size=nr_vertical_bars)
    idx_horiz = np.random.choice(np.arange(height-lgh), replace=False, size=nr_vertical_bars)
    for i in range(nr_vertical_bars):
        img[idx_horiz[i]:idx_horiz[i] + lgh, idx_vert[i]] += 1.
    k = 1
    for i in range(nr_vertical_bars):
        grp[idx_horiz[i]:idx_horiz[i] + lgh, idx_vert[i]] = k
        k += 1

    idx_vert = np.random.choice(np.arange(width- lgh), replace=False, size=nr_horizontal_bars)
    idx_horiz = np.random.choice(np.arange(height), replace=False, size=nr_horizontal_bars)
    for i in range(nr_horizontal_bars):
        img[idx_horiz[i], idx_vert[i]:idx_vert[i] + lgh] = 1.
    for i in range(nr_horizontal_bars):
        grp[idx_horiz[i], idx_vert[i]:idx_vert[i] + lgh] = k
        k += 1


    # idx_horiz = np.random.choice(np.arange(height), replace=False, size=nr_horizontal_bars)
    # img[idx_horiz, :] += 1.
    # for i in idx_horiz:
    #     grp[i, :] = k
    #     k += 1

    grp[img > 1] = 0  # 交叉处颜色为0
    img = img != 0

    return img, grp, grp


def generate_T_segments(width, height, lgh, nr_up_T, nr_down_T):
    """
    generate test bars
    input:
        width: image width
        height: image height
        nr_horizontal_bars: numbers of horizontal bars
        nv_vertical_bars: numbers of vertical bars
    output:
        img: result image
        grp: mask matrix with 1 been selected; 0, not selected
    """
    img = np.zeros((height, width), dtype=np.float)
    grp = np.zeros_like(img)
    grp_p = np.zeros_like(img)

    x_up_T = np.random.choice(np.arange(width-lgh), replace=False, size=nr_up_T)
    y_up_T = np.random.choice(np.arange(height - lgh), replace=False, size=nr_up_T)
    l=1
    for i in range(nr_up_T):
        img[x_up_T[i]+1:x_up_T[i] + lgh, y_up_T[i]+ lgh//2] += 1.
        grp_p[x_up_T[i]+1:x_up_T[i] + lgh, y_up_T[i]+ lgh//2] = l
        l+=1
        img[x_up_T[i], y_up_T[i]: y_up_T[i]+lgh] += 1.
        grp_p[x_up_T[i], y_up_T[i]: y_up_T[i]+lgh] = l
        grp_p[x_up_T[i], y_up_T[i]+ lgh//2] = 0
        l+=1

    k = 1
    for i in range(nr_up_T):
        grp[x_up_T[i]:x_up_T[i] + lgh, y_up_T[i]+ lgh//2] = k
        grp[x_up_T[i], y_up_T[i]: y_up_T[i]+lgh] = k
        k += 1

    x_down_T = np.random.choice(np.arange(width - lgh), replace=False, size=nr_down_T)
    y_down_T = np.random.choice(np.arange(height - lgh), replace=False, size=nr_down_T)

    for i in range(nr_down_T):
        img[x_down_T[i]:x_down_T[i] + lgh-1, y_down_T[i]+lgh//2] += 1.
        grp_p[x_down_T[i]:x_down_T[i] + lgh-1, y_down_T[i]+lgh//2] = l
        l+=1
        img[x_down_T[i] + lgh-1, y_down_T[i]: y_down_T[i] + lgh] += 1.
        grp_p[x_down_T[i] + lgh - 1, y_down_T[i]: y_down_T[i] + lgh] = l
        grp_p[x_down_T[i] + lgh - 1, y_down_T[i]+lgh//2] = 0
        l += 1

    for i in range(nr_down_T):
        grp[x_down_T[i]:x_down_T[i] + lgh, y_down_T[i]+lgh//2] = k
        grp[x_down_T[i] + lgh-1, y_down_T[i]: y_down_T[i] + lgh] = k
        k += 1

    grp[img > 1] = 0  # 交叉处颜色为0
    grp_p[img > 1] = 0
    img = img != 0

    return img, grp, grp_p

if __name__ == "__main__":
    fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(16,5))
    for ax in axes.T:
        img, grp, grp_p = generate_bar_segments(40, 40, 5, 3, 3)
        # plot_groups(grp, ax[1])
        plot_input_image(img, ax[0])
        plot_groups(grp, ax[1])
        plot_groups(grp_p, ax[2])
    plt.savefig('bars_example.png')

    fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(16, 5))
    for ax in axes.T:
        img, grp, grp_p = generate_T_segments(40, 40, 5, 3, 3)
        # plot_groups(grp, ax[1])
        plot_input_image(img, ax[0])
        plot_groups(grp, ax[1])
        plot_groups(grp_p, ax[2])
    plt.savefig('T_example.png')

    # 设置生成相关参数
    np.random.seed(471958)
    nr_train_examples = 60000
    nr_test_examples = 10000
    nr_single_examples = 60000
    width = 40
    height = 40
    lgh = 5
    nr_vert = 6
    nr_horiz= 6
    nr_up=3
    nr_down=3

    # 生成训练集
    data = np.zeros((nr_train_examples, height, width), dtype=np.float32)
    grps = np.zeros_like(data)
    grps2=np.zeros_like(data)
    for i in range(nr_train_examples):
        # data[i, :, :], grps[i, :, :] = generate_bar_segments(width, height, lgh, nr_horiz, nr_vert)
        data[i, :, :], grps[i, :, :], grps2[i, :, :] = generate_bar_segments(width, height, lgh, nr_horiz, nr_vert)

    # 生成训练集
    Tdata = np.zeros((nr_train_examples, height, width), dtype=np.float32)
    Tgrps = np.zeros_like(data)
    Tgrps2 = np.zeros_like(data)
    for i in range(nr_train_examples):
        # Tdata[i, :, :], Tgrps[i, :, :] = generate_T_segments(width, height, lgh, nr_up, nr_down)
        Tdata[i, :, :], Tgrps[i, :, :], Tgrps2[i, :, :] = generate_T_segments(width, height, lgh, nr_up, nr_down)

    # 生成测试集
    test_data = np.zeros((nr_test_examples, height, width), dtype=np.float32)
    test_grps = np.zeros_like(test_data)
    test_grps2 = np.zeros_like(test_data)
    for i in range(nr_test_examples):
        # test_data[i, :, :], test_grps[i, :, :] = generate_bar_segments(width, height, lgh, nr_horiz, nr_vert)
        test_data[i, :, :], test_grps[i, :, :], test_grps2[i, :, :] = generate_bar_segments(width, height, lgh, nr_horiz, nr_vert)

    # 生成测试集
    Ttest_data = np.zeros((nr_test_examples, height, width), dtype=np.float32)
    Ttest_grps = np.zeros_like(test_data)
    Ttest_grps2 = np.zeros_like(test_data)
    for i in range(nr_test_examples):
        # Ttest_data[i, :, :], Ttest_grps[i, :, :] = generate_T_segments(width, height, lgh, nr_up, nr_down)
        Ttest_data[i, :, :], Ttest_grps[i, :, :], Ttest_grps2[i, :, :] = generate_T_segments(width, height, lgh, nr_up, nr_down)

    # 生成只有一条的segment数据（一半竖直，一半水平）
    single_data = np.zeros((nr_single_examples, height, width), dtype=np.float32)
    single_grps = np.zeros_like(single_data)
    for i in range(nr_single_examples // 2):
        single_data[i, :, :], single_grps[i, :, :],_ = generate_bar_segments(width, height, lgh, 1, 0)
    for i in range(nr_single_examples // 2, nr_single_examples):
        single_data[i, :, :], single_grps[i, :, :],_ = generate_bar_segments(width, height, lgh, 0, 1)
    # 保存只含有横的和只含有纵的分别的数据集
    single_data_hori = copy.deepcopy(single_data[:nr_single_examples//2])
    single_grps_hori = copy.deepcopy(single_grps[:nr_single_examples//2])
    single_data_vert = copy.deepcopy(single_data[nr_single_examples//2:])
    single_grps_vert = copy.deepcopy(single_grps[nr_single_examples//2:])
    # 将两组的数据打乱
    shuffel_idx = np.arange(nr_single_examples)
    np.random.shuffle(shuffel_idx)
    print(shuffel_idx)
    single_data = single_data[shuffel_idx, :]
    single_grps = single_grps[shuffel_idx, :]


    # ----------------------------------------------------------------------------


    # 生成只有一条的T（一半up，一半down）
    single_T_data = np.zeros((nr_single_examples, height, width), dtype=np.float32)
    single_T_grps = np.zeros_like(single_T_data)
    single_T_grps2 = np.zeros_like(single_T_data)
    for i in range(nr_single_examples // 2):
        # single_T_data[i, :, :], single_T_grps[i, :, :] = generate_T_segments(width, height, lgh, 1, 0)
        single_T_data[i, :, :], single_T_grps[i, :, :], single_T_grps2[i, :, :] = generate_T_segments(width, height, lgh, 1, 0)
    for i in range(nr_single_examples // 2, nr_single_examples):
        # single_T_data[i, :, :], single_T_grps[i, :, :] = generate_T_segments(width, height, lgh, 0, 1)
        single_T_data[i, :, :], single_T_grps[i, :, :], single_T_grps2[i, :, :] = generate_T_segments(width, height, lgh, 0,1)
    # 保存只含有横的和只含有纵的分别的数据集
    single_T_data_hori = copy.deepcopy(single_T_data[:nr_single_examples // 2])
    single_T_grps_hori = copy.deepcopy(single_T_grps[:nr_single_examples // 2])
    single_T_grps2_hori = copy.deepcopy(single_T_grps2[:nr_single_examples // 2])
    single_T_data_vert = copy.deepcopy(single_T_data[nr_single_examples // 2:])
    single_T_grps_vert = copy.deepcopy(single_T_grps[nr_single_examples // 2:])
    single_T_grps2_vert = copy.deepcopy(single_T_grps2[nr_single_examples // 2:])
    # 将两组的数据打乱
    shuffel_idx = np.arange(nr_single_examples)
    np.random.shuffle(shuffel_idx)
    print(shuffel_idx)
    single_T_data = single_T_data[shuffel_idx, :]
    single_T_grps = single_T_grps[shuffel_idx, :]
    single_T_grps2 = single_T_grps2[shuffel_idx, :]



    data_dir = "../tmp_data/"
    with h5py.File(os.path.join(data_dir, 'bars_part.h5'), 'w') as f:

        single = f.create_group('train_single')
        single.create_dataset('default', data=single_data, compression='gzip', chunks=(100, height, width))
        single.create_dataset('groups1', data=single_grps, compression='gzip', chunks=(100, height, width))
        single.create_dataset('groups2', data=single_grps, compression='gzip', chunks=(100, height, width))

        # singleT = f.create_group('train_T_single')
        # singleT.create_dataset('default', data=single_T_data, compression='gzip', chunks=(100, height, width))
        # singleT.create_dataset('groups', data=single_T_grps, compression='gzip', chunks=(100, height, width))

        train = f.create_group('train_multi')
        train.create_dataset('default', data=data, compression='gzip', chunks=(100, height, width))
        # train.create_dataset('groups', data=grps, compression='gzip', chunks=(100, height, width))
        train.create_dataset('groups1', data=grps, compression='gzip', chunks=(100, height, width))
        train.create_dataset('groups2', data=grps2, compression='gzip', chunks=(100, height, width))

        test = f.create_group('test')
        test.create_dataset('default', data=test_data, compression='gzip', chunks=(100, height, width))
        # test.create_dataset('groups', data=test_grps, compression='gzip', chunks=(100, height, width))
        test.create_dataset('groups1', data=test_grps, compression='gzip', chunks=(100, height, width))
        test.create_dataset('groups2', data=test_grps2, compression='gzip', chunks=(100, height, width))

        # single_hori = f.create_group('train_single_hori')
        # single_hori.create_dataset('default', data=single_data_hori, compression='gzip', chunks=(100, height, width))
        # single_hori.create_dataset('groups', data=single_grps_hori, compression='gzip', chunks=(100, height, width))
        # single_vert = f.create_group('train_single_vert')
        # single_vert.create_dataset('default', data=single_data_vert, compression='gzip', chunks=(100, height, width))
        # single_vert.create_dataset('groups', data=single_grps_vert, compression='gzip', chunks=(100, height, width))

        # single_hori = f.create_group('train_T_single_hori')
        # single_hori.create_dataset('default', data=single_T_data_hori, compression='gzip', chunks=(100, height, width))
        # single_hori.create_dataset('groups', data=single_T_grps_hori, compression='gzip', chunks=(100, height, width))
        # single_vert = f.create_group('train_T_single_vert')
        # single_vert.create_dataset('default', data=single_T_data_vert, compression='gzip', chunks=(100, height, width))
        # single_vert.create_dataset('groups', data=single_T_grps_vert, compression='gzip', chunks=(100, height, width))

    with h5py.File(os.path.join(data_dir, 'bars_whole.h5'), 'w') as f:
        # single = f.create_group('train_single')
        # single.create_dataset('default', data=single_data, compression='gzip', chunks=(100, height, width))
        # single.create_dataset('groups', data=single_grps, compression='gzip', chunks=(100, height, width))
        singleT = f.create_group('train_single')
        singleT.create_dataset('default', data=single_T_data, compression='gzip', chunks=(100, height, width))
        # singleT.create_dataset('groups', data=single_T_grps, compression='gzip', chunks=(100, height, width))
        singleT.create_dataset('groups1', data=single_T_grps, compression='gzip', chunks=(100, height, width))
        singleT.create_dataset('groups2', data=single_T_grps2, compression='gzip', chunks=(100, height, width))

        train = f.create_group('train_multi')
        train.create_dataset('default', data=Tdata, compression='gzip', chunks=(100, height, width))
        # train.create_dataset('groups', data=Tgrps, compression='gzip', chunks=(100, height, width))
        train.create_dataset('groups1', data=Tgrps, compression='gzip', chunks=(100, height, width))
        train.create_dataset('groups2', data=Tgrps2, compression='gzip', chunks=(100, height, width))

        test = f.create_group('test')
        test.create_dataset('default', data=Ttest_data, compression='gzip', chunks=(100, height, width))
        # test.create_dataset('groups', data=Ttest_grps, compression='gzip', chunks=(100, height, width))
        test.create_dataset('groups1', data=Ttest_grps, compression='gzip', chunks=(100, height, width))
        test.create_dataset('groups2', data=Ttest_grps2, compression='gzip', chunks=(100, height, width))

        # single_hori = f.create_group('train_single_hori')
        # single_hori.create_dataset('default', data=single_data_hori, compression='gzip', chunks=(100, height, width))
        # single_hori.create_dataset('groups', data=single_grps_hori, compression='gzip', chunks=(100, height, width))
        # single_vert = f.create_group('train_single_vert')
        # single_vert.create_dataset('default', data=single_data_vert, compression='gzip', chunks=(100, height, width))
        # single_vert.create_dataset('groups', data=single_grps_vert, compression='gzip', chunks=(100, height, width))

        # single_hori = f.create_group('train_single_hori')
        # single_hori.create_dataset('default', data=single_T_data_hori, compression='gzip', chunks=(100, height, width))
        # single_hori.create_dataset('groups', data=single_T_grps_hori, compression='gzip', chunks=(100, height, width))
        # single_vert = f.create_group('train_single_vert')
        # single_vert.create_dataset('default', data=single_T_data_vert, compression='gzip', chunks=(100, height, width))
        # single_vert.create_dataset('groups', data=single_T_grps_vert, compression='gzip', chunks=(100, height, width))
