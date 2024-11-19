import os
import copy
import numpy as np
import networkx as nx
from time import perf_counter as t
import random
import torch

def drop_feature(x, drop_prob):
    # 使节点的某一维度为0
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
