#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedWeightAvg(w, size):
    # 加权Fed：数据量越多，权重越大
    totalSize = sum(size)
    # 取出第一个client的参数
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys(): # 对应的网络结构名称和参数
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        # 每个网络结构的参数都进行一下加权平均
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg