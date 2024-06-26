import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

def gen_counting_label(labels, channel, tag):
    labels = labels.flatten(1)
    batch_size, labels_lenght = labels.size()
    device = labels.device
    counting_labels = torch.zeros((batch_size, channel))
    if tag:
        ignore = [0, 156, 157, 184, 186] # ignores some labels for counting as they are invisible
    else:
        ignore = []
    for i in range(batch_size):
        for j in range(labels_lenght):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    return counting_labels.to(device)
