# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch

# Add to datautil/util.py
def act_param_init(args):
    args.select_position = {'emg': [0]}
    args.select_channel = {'emg': np.arange(8)}
    args.hz_list = {'emg': 1000}
    args.act_people = {'emg': [[i*9+j for j in range(9)]for i in range(4)]}
    tmp = {'emg': ((8, 1, 200), 6, 10)}
    args.num_classes, args.input_shape, args.grid_size = tmp[
        args.dataset][1], tmp[args.dataset][0], tmp[args.dataset][2]

    return args

def get_dataset(args):
    """Return train/val/test splits based on args.dataset and args.task."""
    # --- Example for EMG ---
    from .emg_dataset import EMGDataset  # adjust to your code  
    train_set = EMGDataset(args, split='train')
    val_set = EMGDataset(args, split='val')
    test_set = EMGDataset(args, split='test')
    return train_set, val_set, test_set

def get_input_shape(dataset):
    """Extract input_shape from dataset."""
    sample = dataset[0][0]  # x, y, d, pc, pd, idx
    return sample.shape

# Ensure Nmax is properly defined as you have it

def Nmax(args, d):
    for i in range(len(args.test_envs)):
        if d < args.test_envs[i]:
            return i
    return len(args.test_envs)

class basedataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class mydataset(object):
    def __init__(self, args):
        self.x = None
        self.labels = None        # Class labels
        self.dlabels = None       # Domain labels
        self.pclabels = None
        self.pdlabels = None
        self.task = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None
        self.args = args

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'pclabel':
            self.pclabels = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels = tlabels
        elif label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif label_type == 'domain_label':
            self.dlabels[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.x[index])

        ctarget = self.target_trans(self.labels[index]) if self.labels is not None else None
        # --- Add robust domain label support for curriculum/K ---
        dtarget = self.target_trans(self.dlabels[index]) if self.dlabels is not None else -1
        pctarget = self.target_trans(self.pclabels[index]) if self.pclabels is not None else None
        pdtarget = self.target_trans(self.pdlabels[index]) if self.pdlabels is not None else None

        # Always return x, class label, domain label, and others (order is important)
        return x, ctarget, dtarget, pctarget, pdtarget, index

    def __len__(self):
        return len(self.x)

    # New: Utility to expose only domain labels (for clustering/curriculum)
    def get_domain_labels(self):
        return self.dlabels if self.dlabels is not None else np.zeros(len(self.x), dtype=int)

class subdataset(mydataset):
    def __init__(self, args, dataset, indices):
        super(subdataset, self).__init__(args)
        self.x = dataset.x[indices]
        self.loader = dataset.loader
        self.labels = dataset.labels[indices] if dataset.labels is not None else None
        self.dlabels = dataset.dlabels[indices] if dataset.dlabels is not None else None
        self.pclabels = dataset.pclabels[indices] if dataset.pclabels is not None else None
        self.pdlabels = dataset.pdlabels[indices] if dataset.pdlabels is not None else None
        self.task = dataset.task
        self.dataset = dataset.dataset
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

class combindataset(mydataset):
    def __init__(self, args, datalist):
        super(combindataset, self).__init__(args)
        self.domain_num = len(datalist)
        self.loader = datalist[0].loader
        xlist = [item.x for item in datalist]
        cylist = [item.labels for item in datalist]
        dylist = [item.dlabels for item in datalist]
        pcylist = [item.pclabels for item in datalist]
        pdylist = [item.pdlabels for item in datalist]
        self.dataset = datalist[0].dataset
        self.task = datalist[0].task
        self.transform = datalist[0].transform
        self.target_transform = datalist[0].target_transform
        self.x = torch.vstack(xlist)

        self.labels = np.hstack(cylist) if cylist[0] is not None else None
        self.dlabels = np.hstack(dylist) if dylist[0] is not None else None
        self.pclabels = np.hstack(pcylist) if pcylist[0] is not None else None
        self.pdlabels = np.hstack(pdylist) if pdylist[0] is not None else None
