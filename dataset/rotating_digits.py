from .dataset_base import BaseDataset
from torch.utils.data import DataLoader
import numpy as np
import os

class VariedRotatingDigitsOnlinegen(BaseDataset):
    def __init__(self, directory, yp_window_len, y_window_len, x_window_len, window_stride, 
                       num_seq_dataset=500, one_hot_label=False, seed=0, switch_samples_per_sequence=300, switch_probability = .2,**kwargs): #switch_probability = 0.2
        super().__init__()
        # each sequence is a different mnist digit rotated through seq_len many angles
        all_Y = np.load(os.path.join(directory, 'images_data.npy'),mmap_mode='r') 
        all_X = np.load(os.path.join(directory, 'digits_class.npy'),mmap_mode='r')
        num_seq, seq_len, ch, res1, res2 = all_Y.shape

        assert res1 == res2
        assert ch == 1 # grey level
        res = res1
        assert all_X.shape[0] == num_seq and all_X.shape[1] == seq_len

        self.seed = seed
        self.y_window_len = y_window_len
        self.yp_window_len = yp_window_len
        self.window_stride = window_stride
        self.switch_samples_per_sequence = switch_samples_per_sequence
        self.switch_probability = switch_probability

        #hard code to only take the first num_seq_dataset sequences for processing for memory. TODO make the dataset in memory friendly format
        
        num_seq=min(num_seq,num_seq_dataset)
        self.all_Y = all_Y[:num_seq]
        self.all_X = all_X[:num_seq]

        #arranging the sequences
        sample_len = y_window_len + yp_window_len
        cnt_in_seq = (seq_len - sample_len) // window_stride + 1
        self.sample_len = sample_len
        self.cnt_in_seq = cnt_in_seq

        self.num_seq = num_seq

    def __getitem__(self, idx):
        # an index has to enumerate: which sequence, which strided window in that sequence, and which sample for the last two (switch_samples_per_sequence)
        seq = idx % self.num_seq
        stridepos = idx % self.cnt_in_seq

        startseq = stridepos*self.window_stride
        endseq = stridepos*self.window_stride + self.y_window_len
        endseq_p = stridepos*self.window_stride + self.sample_len

        window_img = np.copy(self.all_Y[seq, startseq: endseq_p])
        window_lbl = np.copy(self.all_X[seq, startseq: endseq_p]) # note that dataset it works with is no switch ones, so cons label
        
        # need to create a local random generator so as to not change the global seed
        generator = np.random.default_rng(seed=idx)
        if generator.random() <= self.switch_probability:
            #switches this digit with another in the same rotation angle
            switched_seq_idx = generator.integers(0,self.num_seq)
            window_img[self.y_window_len : self.sample_len] = self.all_Y[switched_seq_idx, endseq:endseq_p]
            window_lbl[self.y_window_len - 1 : self.sample_len] = self.all_X[switched_seq_idx, endseq-1:endseq_p]


        return window_img[self.y_window_len : self.sample_len], window_img[:self.y_window_len], window_lbl

    def __len__(self):
        # an index has to enumerate: which sequence, which strided window in that sequence, and which sample for the last two (switch_samples_per_sequence)
        # therefore length is num_seq*cnt_in_seq*switch_samples_per_sequence

        return self.num_seq*self.cnt_in_seq*self.switch_samples_per_sequence