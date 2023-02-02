#!/bin/bash 
python train.py ALDA --gpu_id 1 --net ResNet50 --add_method BNM --batch_size 20 --trade_off 1 --output_dir "OR" --loss_type all --threshold 0.9