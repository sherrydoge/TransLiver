#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29500 train.py --transform --pretrain
