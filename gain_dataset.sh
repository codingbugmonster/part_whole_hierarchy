#!/bin/bash

if [ ! -d "./tmp_data" ]; then
    mkdir ./tmp_data
fi

python ./dataset/Ts.py
echo "finish Ts"

python ./dataset/Squares.py
echo "finish Squares"

python ./dataset/double_MNIST.py
echo "finish double_MNIST"

python ./dataset/SHOPs.py
echo "finish SHOPs"

