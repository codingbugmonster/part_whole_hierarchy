#!/bin/bash

if [ ! -d "./tmp_net" ]; then
    mkdir ./tmp_net
fi

if [ ! -d "./tmp_img" ]; then
    mkdir ./tmp_img
fi

python ./train_nets/train_SHOPs.py
echo "finish SHOPs net"

python ./train_nets/train_Squares.py
echo "finish Squares net"

python ./train_nets/train_Ts.py
echo "finish Ts net"

python ./train_nets/train_Double_MNIST.py
echo "finish Double_MNIST net"

