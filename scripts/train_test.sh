#!/usr/bin/env bash

python train.py --name 'test_labelup_train' --dataroot 'datasets/overfit_train/' \
    --lod_train_img 4 --lod_transition_img 4 --no_instance