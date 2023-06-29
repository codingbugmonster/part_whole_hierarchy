from numpy.core.numeric import Inf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import time
import matplotlib.pyplot as plt
import math

import dataset.plot_tools as plot_tools
import opensource.dae as dae
import dataset_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_scheduler(optimizer):
    for param_group in optimizer.param_groups:
       param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def salt_pepper_noise(X, p=0.5):
    mask = torch.rand(X.shape, dtype=torch.float32)
    mask = (mask >= 0.6) #0.6,0.4
    X = mask * X
    return X


def draw_fig(iter, X, label, fig_num, H, W, dir, channel_num=2):
    X = X.cpu().detach().numpy()
    label = label.cpu().numpy()
    ncols = 6
    nrows = math.ceil(fig_num/6)
    if channel_num == 3:
        subfig_num = X.shape[0]
        width = math.ceil(math.sqrt(subfig_num))
        height = math.ceil(subfig_num / width)

        fig = plt.figure()
        idx = 0
        for i in range(height):
            for j in range(width):
                fig.add_subplot(height*2, width, i*width+j+1)
                img = X[idx].reshape(3, H, W).transpose(1, 2, 0)
                plt.imshow(img)
                idx += 1
                
        idx = 0
        for i in range(height):
            for j in range(width):
                fig.add_subplot(height*2, width, (i+height)*width+j+1)
                img = label[idx].reshape(3, H, W).transpose(1, 2, 0)
                plt.imshow(img)
                idx += 1
        plt.savefig(dir + str(iter) + '.png')
        return 

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows * 2, figsize=(16, 5*nrows))
    
    for i in range(nrows * 2):
        for j in range(ncols):
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['bottom'].set_visible(False)
            axes[i][j].spines['left'].set_visible(False)

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            # print(X[idx].reshape(H, W))
            plot_tools.plot_input_image(X[idx].reshape(H, W), axes[i][j])
            idx += 1
            if idx >= fig_num:
                break

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            plot_tools.plot_input_image(label[idx].reshape(H, W), axes[i + nrows][j])
            idx += 1
            if idx >= fig_num:
                break

    fig.savefig(dir + str(iter) + '.png')


def train_dae(net, save_dir, dataset_name, H, W, batch_size=16, num_epoch=200, lr=0.1,
    log_iter=False, max_unchange_epoch=20, fig_dir='./tmp_img/bars_'):
    """
    Train DAE network.

    Inputs:
        batch_size: batch size
        num_epoch: maximal epoch
        lr: initial learning rate
        log_iter: whether log info of each iteration in one epoch
        max_unchange_epoch: maximal epoch number unchanged verification loss, exceed will stop training 
    Outputs:
    """
    train_dataset = dataset_utils.BindingDataset("../tmp_data", dataset_name, is_single=True,
                                                    train=True, ver=True, is_ver=False)
    val_dataset = dataset_utils.BindingDataset("../tmp_data", dataset_name, is_single=True,
                                                    train=True, ver=True, is_ver=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    ver_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    
    
    creterion = F.binary_cross_entropy
    # creterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_ver_loss = Inf
    unchange_epoch = 0

    for epoch in range(num_epoch):
        running_loss = 0
        total_loss = 0

        for iter, (X, _) in enumerate(train_loader):
            # net.zero_grad()
            
            X = X.reshape(X.shape[0], -1)
            label = copy.deepcopy(X)  # reconstruct
            # label = label.type(torch.long)
            X = salt_pepper_noise(X)

            X = X.to(device)
            label = label.to(device)

            output = net(X)
            # print(output[0].reshape(20, 20))

            loss = creterion(output, label)
            running_loss += loss.cpu().item()
            total_loss += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter % 10 == 0) and log_iter:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epoch, iter+1, len(train_dataset)//batch_size, running_loss))
                running_loss = 0
            
            if iter == 0 and epoch % 10 == 0 :#and epoch != 0:
                if dataset_name == "clevr":
                    draw_fig(epoch, output, X, batch_size, H, W, dir=fig_dir, channel_num=3)
                else:
                    draw_fig(epoch, output, X, batch_size, H, W, dir=fig_dir)

        # 更新学习率
        if epoch == 60:
            optimizer = lr_scheduler(optimizer)   
        if epoch == 150:
            optimizer = lr_scheduler(optimizer)     

        # 输出EPOCH训练信息
        print("After training epoch [%d], loss [%.5f]" % (epoch, total_loss))

        # 验证集
        with torch.no_grad():
            cur_ver_loss = 0
            for iter, (X, _) in enumerate(ver_loader):
                X = X.reshape(X.shape[0], -1)
                label = copy.deepcopy(X)
                
                X = X.to(device)
                label = label.to(device)

                output = net(X)
                loss = creterion(output, label)
                cur_ver_loss += loss.cpu().item()

                # if iter == 0 and epoch % 50 == 0 and epoch != 0:
                #     draw_fig(epoch, output, label, batch_size, H, W, dir=fig_dir)
            
            if cur_ver_loss < min_ver_loss:
                min_ver_loss = cur_ver_loss
                unchange_epoch = 0
            else:
                unchange_epoch += 1
        
        # 输出EPOCH验证集信息
        print("After verification epoch [%d], loss [%.5f, %.5f]" % (epoch, cur_ver_loss, min_ver_loss))

        # if unchange_epoch > max_unchange_epoch:
        #     break

    torch.save(net, save_dir)

if __name__ == "__main__":

    H = 60
    W = 60

    net1 = dae.dae(H * W, 400).to(device)
    train_dae(net1, save_dir="../tmp_net/corner_part_net2_08.pty", dataset_name="corners_part", H=H, W=W,
                log_iter=False, fig_dir='../tmp_img/corners_part_',lr=0.001)
    net2 = dae.dae(H * W, 400).to(device)
    train_dae(net2, save_dir="../tmp_net/corner_whole_net2_06.pty", dataset_name="corners_whole", H=H, W=W,
              log_iter=False, fig_dir='../tmp_img/corners_whole_',lr = 0.001)


