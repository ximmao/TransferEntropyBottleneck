from .dataset_base import BaseDataset

import torch.nn as nn
import torch
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
from torch.utils.data import DataLoader
import numpy as np
import os

angularVelo_mul = [0.2, 0.4, 0.6, 0.8, 1.0]

class FrequencyChangingSinesSummedMultiple(BaseDataset):
    def __init__(self, directory, yp_window_len, y_window_len, x_window_len, 
                       num_dataset=30000, seed=0, include_first_in_target=True, input_mean=False, output_mean=True,
                       baseline_train=False, label_noise=0., label_dim=5, split_name='train', **kwargs): #switch_probability = 0.2
        super().__init__()
        # self.seed = seed
        all_Y = np.load(os.path.join(directory, 'signals_data.npy'),mmap_mode='r') 
        all_X = np.load(os.path.join(directory, 'freq_class.npy'),mmap_mode='r')
        num_seq, seq_len, ch = all_Y.shape

        num_seq=min(num_seq,num_dataset)
        self.all_Y = all_Y[:num_seq]
        self.all_X = all_X[:num_seq]
        self.num_seq = num_seq
        
        precompute_len_y = 100
        precompute_len_yp = 20
        self.y_window_len = y_window_len
        self.yp_window_len = yp_window_len
        self.precompute_len_y = precompute_len_y
        self.precompute_len_yp = precompute_len_yp
        assert (self.y_window_len <= precompute_len_y) and (self.yp_window_len <= precompute_len_yp)
        self.num_dataset = num_dataset
        # self.switch_probability = switch_probability
        self.include_first_in_target = include_first_in_target
        self.split_name = split_name
        self.baseline_train = baseline_train
        self.output_mean = output_mean
        self.input_mean = input_mean
        self.label_noise = label_noise
        self.label_dim = int(label_dim)
        assert self.label_dim <= 5
        if self.label_dim < 5 and self.baseline_train == True:
            print('reduced label dimension not compatible with baseline train, ignore the special label dimension')
        # print(label_noise)

    def __getitem__(self, idx):
        # generate the sequence on the fly
        # need to create a local random generator so as to not change the global seed
        # window_lbl = self.all_X[idx, (self.precompute_len_y-self.y_window_len):(self.precompute_len_y+self.yp_window_len)].reshape(self.y_window_len+self.yp_window_len, -1)
        true_x = self.all_Y[idx, (self.precompute_len_y-self.y_window_len):self.precompute_len_y].reshape(self.y_window_len, -1)
        if self.include_first_in_target:
            true_xp = self.all_Y[idx, (self.precompute_len_y-1):(self.precompute_len_y+self.yp_window_len)].reshape(self.yp_window_len+1, -1)
        else:
            true_xp = self.all_Y[idx, self.precompute_len_y:(self.precompute_len_y+self.yp_window_len)].reshape(self.yp_window_len, -1)
        
        if self.output_mean:
            assert len(true_xp.shape) == 2
            assert true_xp.shape[-1] == true_x.shape[-1]
            true_xp = np.mean(true_xp, keepdims=True, axis=-1)
        
        if self.input_mean:
            assert len(true_x.shape) == 2
            true_x = np.mean(true_x, keepdims=True, axis=-1)
        
        if self.label_noise > 0.:
            if self.split_name == 'test':
                possible_length = int(2*(40000.))
                generator = np.random.default_rng(seed=(idx+possible_length))
            elif self.split_name == 'valid':
                possible_length = int(40000.)
                generator = np.random.default_rng(seed=(idx+possible_length))
            else:
                generator = np.random.default_rng(seed=idx)
            added_noise = (generator.random(size=(self.y_window_len+self.yp_window_len, self.all_X.shape[2]))*self.label_noise*2. - self.label_noise)
            window_lbl = self.all_X[idx, (self.precompute_len_y-self.y_window_len):(self.precompute_len_y+self.yp_window_len)].reshape(self.y_window_len+self.yp_window_len, -1) + added_noise
        else:
            window_lbl = self.all_X[idx, (self.precompute_len_y-self.y_window_len):(self.precompute_len_y+self.yp_window_len)].reshape(self.y_window_len+self.yp_window_len, -1)
        assert window_lbl.shape[-1] == 5

        if self.baseline_train:
            return true_xp, np.concatenate((true_x, window_lbl[:self.y_window_len]), -1), window_lbl
        else:
            return true_xp, true_x, window_lbl[:, :self.label_dim]

    def __len__(self):
        # an index has to enumerate: which sequence, which strided window in that sequence, and which sample for the last two (switch_samples_per_sequence)
        # therefore length is num_seq*cnt_in_seq*switch_samples_per_sequence

        return self.num_seq