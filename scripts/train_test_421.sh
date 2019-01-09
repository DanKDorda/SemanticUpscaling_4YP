#!/usr/bin/env bash

python train.py --name 'test_labelup_train_421_3' --dataroot 'datasets/overfit_train/' \
    --batchSize 2 --niter 150 --niter_decay 150 --lod_train_img 1000 --lod_transition_img 1000 --num_phases 2 --no_vgg_loss --no_instance