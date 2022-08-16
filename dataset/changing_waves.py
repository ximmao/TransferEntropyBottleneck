from .dataset_base import BaseDataset
from torch.utils.data import DataLoader
import numpy as np
import os

from torchdiffeq import odeint_adjoint as odeint

angularVelo_mul = [0.2, 0.4, 0.6, 0.8, 1.0]

class TrueFunc(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A

    def forward(self, t, x):
        return torch.matmul(x, self.A)        

def get_A(k,fs=20.):
    return torch.Tensor([[0., -2.*k*np.pi/fs], [2.*k*np.pi/fs, 0.]])

class FreqChangingSinesOnlinegen(BaseDataset):
    def __init__(self, directory, yp_window_len, y_window_len, x_window_len, 
                       num_dataset=60000, seed=0, switch_probability = .2, 
                       precompute_len_y = 100, precompute_len_yp = 51, **kwargs): #switch_probability = 0.2
        super().__init__()
        # self.seed = seed
        self.y_window_len = y_window_len
        self.yp_window_len = yp_window_len
        self.precompute_len_y = precompute_len_y
        self.precompute_len_yp = precompute_len_yp
        assert (self.y_window_len <= precompute_len_y) and (self.yp_window_len <= (precompute_len_yp-1))
        self.num_dataset = num_dataset
        self.switch_probability = switch_probability

    def __getitem__(self, idx):
        # generate the sequence on the fly
        # need to create a local random generator so as to not change the global seed
        generator = np.random.default_rng(seed=idx)
        am = generator.integers(len(angularVelo_mul))
        true_A = get_A(angularVelo_mul[am])
        odetrue = TrueFunc(true_A)
        t = torch.linspace(0., self.precompute_len_y, self.precompute_len_y)
        window_lbl = torch.zeros(self.y_window_len) + float(am)
        w = float(generator.integers(36))
        true_x0 = torch.Tensor([np.cos(w*np.pi/36.), np.sin(w*np.pi/36.)])
        true_x = odeint(odetrue, true_x0, t, method='rk4')
        tp = torch.linspace(self.precompute_len_y, (self.precompute_len_y+self.precompute_len_yp), self.precompute_len_yp)
        if generator.random() <= self.switch_probability:
            amp = generator.integers(len(angularVelo_mul))
            window_lbl[-1] = float(amp)
            true_Ap = get_A(angularVelo_mul[amp]) 
            odetruep = TrueFunc(true_Ap)
            true_xp = odeint(odetruep, true_x[-1], tp, method='rk4')
            # switched_seq_idx = generator.integers(0,self.num_seq)
            # window_img[self.y_window_len : self.sample_len] = self.all_Y[switched_seq_idx, endseq:endseq_p]
            # window_lbl[self.y_window_len - 1 : self.sample_len] = self.all_X[switched_seq_idx, endseq-1:endseq_p]
        else:
            true_xp = odeint(odetrue, true_x[-1], tp, method='rk4')


        return true_xp[:,1][1:(self.y_window_len+1)].unsqueeze(-1), true_x[:,1][(-self.y_window_len):].unsqueeze(-1), window_lbl.unsqueeze(-1)

    def __len__(self):
        # an index has to enumerate: which sequence, which strided window in that sequence, and which sample for the last two (switch_samples_per_sequence)
        # therefore length is num_seq*cnt_in_seq*switch_samples_per_sequence

        return self.num_dataset