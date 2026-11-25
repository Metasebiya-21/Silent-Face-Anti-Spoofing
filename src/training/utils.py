# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午2:13
# @Author : zhuying
# @Company : Minivision
# @File : utility.py
# @Software : PyCharm

from datetime import datetime
import os


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
