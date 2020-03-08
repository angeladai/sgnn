import os, sys, struct
import scipy.io as sio
import numpy as np
import torch
import torch.utils.data
import random
import math
import plyfile

import data_util
import marching_cubes as mc

def collate(batch):
    names = [x['name'] for x in batch]
    # collect sparse inputs
    locs = batch[0]['input'][0]
    locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).long()], 1)
    feats = batch[0]['input'][1]
    known = None
    if batch[0]['known'] is not None:
        known = torch.stack([x['known'] for x in batch])
    colors = None
    hierarchy = None
    if batch[0]['hierarchy'] is not None:
        hierarchy = [None]*len(batch[0]['hierarchy'])
        for h in range(len(batch[0]['hierarchy'])):
            hierarchy[h] = torch.stack([x['hierarchy'][h] for x in batch])
    for b in range(1, len(batch)):
        cur_locs = batch[b]['input'][0]
        cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long()*b], 1)
        locs = torch.cat([locs, cur_locs])
        feats = torch.cat([feats, batch[b]['input'][1]])
    sdfs = torch.stack([x['sdf'] for x in batch])
    world2grids = torch.stack([x['world2grid'] for x in batch])
    orig_dims = torch.stack([x['orig_dims'] for x in batch])
    return {'name': names, 'input': [locs,feats], 'sdf': sdfs, 'world2grid': world2grids, 'known': known, 'hierarchy': hierarchy, 'orig_dims': orig_dims}


class SceneDataset(torch.utils.data.Dataset):

    def __init__(self, files, input_dim, truncation, num_hierarchy_levels, max_input_height, num_overfit=0, target_path=''):
        assert(num_hierarchy_levels <= 4) # havent' precomputed more than this
        self.is_chunks = target_path == '' # have target path -> full scene data
        if not target_path:
            self.files = [f for f in files if os.path.isfile(f)]
        else:
            self.files = [(f,os.path.join(target_path, os.path.basename(f))) for f in files if (os.path.isfile(f) and os.path.isfile(os.path.join(target_path, os.path.basename(f))))]
        self.input_dim = input_dim
        self.truncation = truncation
        self.num_hierarchy_levels = num_hierarchy_levels
        self.max_input_height = max_input_height
        self.UP_AXIS = 0
        if num_overfit > 0:
            num_repeat = max(1, num_overfit // len(self.files))
            self.files = self.files * num_repeat
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        name = None
        if self.is_chunks:
            name = os.path.splitext(os.path.basename(file))[0]
            
            inputs, targets, dims, world2grid, target_known, target_hierarchy = data_util.load_train_file(file)
        else:
            input_file = file[0]
            target_file = file[1]
            name = os.path.splitext(os.path.basename(input_file))[0]
            
            inputs, dims, world2grid = data_util.load_scene(input_file)
            targets, dims, world2grid = data_util.load_scene(target_file)
            target_known = data_util.load_scene_known(os.path.splitext(target_file)[0] + '.knw')
            targets = data_util.sparse_to_dense_np(targets[0], targets[1][:,np.newaxis], dims[2], dims[1], dims[0], -float('inf'))
            target_hierarchy = None
        
        orig_dims = torch.LongTensor(targets.shape)
        if not self.is_chunks: 
            # add padding
            hierarchy_factor = pow(2, self.num_hierarchy_levels-1)
            max_input_dim = np.array(targets.shape)
            if self.max_input_height > 0 and max_input_dim[self.UP_AXIS] > self.max_input_height:
                max_input_dim[self.UP_AXIS] = self.max_input_height
                mask_input = inputs[0][:,self.UP_AXIS] < self.max_input_height
                inputs[0] = inputs[0][mask_input]
                inputs[1] = inputs[1][mask_input]
            max_input_dim = ((max_input_dim + (hierarchy_factor*4) - 1) // (hierarchy_factor*4)) * (hierarchy_factor*4)
            # pad target to max_input_dim
            padded = np.zeros((max_input_dim[0], max_input_dim[1], max_input_dim[2]), dtype=np.float32)
            padded.fill(-float('inf'))
            padded[:min(self.max_input_height, targets.shape[0]), :targets.shape[1], :targets.shape[2]] = targets[:self.max_input_height, :, :]
            targets = padded
            if target_known is not None:
                known_pad = np.ones((max_input_dim[0], max_input_dim[1], max_input_dim[2]), dtype=np.uint8) * 255
                known_pad[:min(self.max_input_height,target_known.shape[0]), :target_known.shape[1], :target_known.shape[2]] = target_known[:self.max_input_height, :, :]
                target_known = known_pad
        else:
            if self.num_hierarchy_levels < 4:
                target_hierarchy = target_hierarchy[4-self.num_hierarchy_levels:]

        mask = np.abs(inputs[1]) < self.truncation
        input_locs = inputs[0][mask]
        input_vals = inputs[1][mask]
        inputs = [torch.from_numpy(input_locs).long(), torch.from_numpy(input_vals[:,np.newaxis]).float()]

        targets = targets[np.newaxis,:]
        targets = torch.from_numpy(targets)
        if target_hierarchy is not None:
            for h in range(len(target_hierarchy)):
                target_hierarchy[h] = torch.from_numpy(target_hierarchy[h][np.newaxis,:])
        world2grid = torch.from_numpy(world2grid)
        target_known = target_known[np.newaxis,:]
        target_known = torch.from_numpy(target_known)
        sample = {'name': name, 'input': inputs, 'sdf': targets, 'world2grid': world2grid, 'known': target_known, 'hierarchy': target_hierarchy, 'orig_dims': orig_dims}
        return sample




