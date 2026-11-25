# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
Training configuration. Paths resolve from the repo root, so notebooks work from any folder.
"""

from datetime import datetime
from pathlib import Path

import torch
from easydict import EasyDict
from liveness_service.ml.utils import get_kernel
from model_development.utils import get_width_height, make_if_not_exist

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_default_config():
    conf = EasyDict()

    # ----------------------training---------------
    conf.lr = 1e-1
    # [9, 13, 15]
    conf.milestones = [10, 15, 22]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 25
    conf.momentum = 0.9
    conf.batch_size = 1024

    # model
    conf.num_classes = 3
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset: <train_root_path>/<patch_info>/{0,1,2}/*.png
    conf.train_root_path = REPO_ROOT / 'datasets' / 'rgb_image' / 'train'
    conf.num_workers = 4  # DataLoader worker processes

    # save file path
    conf.snapshot_dir_path = REPO_ROOT / 'saved_logs' / 'snapshot'

    # log path
    conf.log_path = REPO_ROOT / 'saved_logs' / 'jobs'
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 30

    return conf


def update_config(conf, patch_info, devices=(0,)):
    """Fill in what depends on the patch type, e.g. patch_info="2.7_80x80"; devices: GPU ids (ignored on CPU)."""
    conf.devices = list(devices)
    conf.patch_info = patch_info
    w_input, h_input = get_width_height(patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(patch_info)
    log_path = Path(conf.log_path) / job_name / current_time
    snapshot_dir = Path(conf.snapshot_dir_path) / job_name

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
