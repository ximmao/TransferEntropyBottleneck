import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib

angularVelo_mul = [0.2, 0.4, 0.6, 0.8, 1.0]

class TrueFunc(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A

    def forward(self, t, x):
        return torch.matmul(x, self.A)        

def get_A(k,fs=20.):
    return torch.Tensor([[0., -2.*k*np.pi/fs], [2.*k*np.pi/fs, 0.]])

def save_npy(cfilename, data):
    with open(cfilename, mode='wb') as f:
        np.save(f, data)

precompute_len_y = 100
precompute_len_yp = 20
switch = True
# label_noise = True
if not switch:
    switch_probability = 0.
    dump_dir='./data/fwaves_data_noswitch_multi/'
else:
    switch_probability = 0.5
    dump_dir='./data/fwaves_data_switch_0p5_multi/'
num_dataset = {'train':30000, 'valid':5000, 'test':5000}
dset = {'train':{'signal':np.zeros((num_dataset['train'], precompute_len_y+precompute_len_yp, len(angularVelo_mul))),
                 'label':np.zeros((num_dataset['train'], precompute_len_y+precompute_len_yp, len(angularVelo_mul)))}, 
        'valid':{'signal':np.zeros((num_dataset['valid'], precompute_len_y+precompute_len_yp, len(angularVelo_mul))),
                 'label':np.zeros((num_dataset['valid'], precompute_len_y+precompute_len_yp, len(angularVelo_mul)))}, 
        'test':{'signal':np.zeros((num_dataset['test'], precompute_len_y+precompute_len_yp, len(angularVelo_mul))),
                 'label':np.zeros((num_dataset['test'], precompute_len_y+precompute_len_yp, len(angularVelo_mul)))}}
for split_name in ['train', 'valid', 'test']:
    cnt_tot=0
    cnt_swt=0
    print('create', split_name)
    for idx in range(num_dataset[split_name]):
        if idx % 500 == 0 or idx == (num_dataset[split_name]-1):
            print(idx)
        if split_name == 'test':
            possible_length = int(2*(num_dataset['train'] + num_dataset['train'] / 3.))
            generator = np.random.default_rng(seed=(idx+possible_length))
        elif split_name == 'valid':
            possible_length = int(num_dataset['train'] + num_dataset['train'] / 3.)
            generator = np.random.default_rng(seed=(idx+possible_length))
        else:
            generator = np.random.default_rng(seed=idx)

        # with or without replace 
        am_arr = generator.choice(np.arange(len(angularVelo_mul)), size=len(angularVelo_mul), replace=False)

        # have switch in this sample?
        idx_swt = None
        if generator.random() <= switch_probability:
            cnt_swt+=1
            idx_swt = generator.integers(len(angularVelo_mul)) # the one to switch
        cnt_tot+=1
        idx_sam = 0
        print(am_arr, 'idx to switch', idx_swt)
        for am in am_arr:

            true_A = get_A(angularVelo_mul[am])
            odetrue = TrueFunc(true_A)
            t = torch.linspace(0., float(precompute_len_y), int(precompute_len_y+1))
            window_lbl = torch.zeros(precompute_len_y+precompute_len_yp) + float(angularVelo_mul[int(am)])
            # w = float(generator.integers(36))
            # true_x0 = torch.Tensor([np.cos(w*2*np.pi/36.), np.sin(w*2*np.pi/36.)])
            
            # Sample initial conditions
            w = float(generator.random())
            true_x0 = torch.Tensor([np.cos(w*2*np.pi), np.sin(w*2*np.pi)])
            true_x = odeint(odetrue, true_x0, t)
            # print(true_x.size())

            tp = torch.linspace(float(precompute_len_y), float(precompute_len_y+precompute_len_yp), int(precompute_len_yp+1))
            if idx_sam == idx_swt:
                amp = generator.integers(len(angularVelo_mul))
                window_lbl[(precompute_len_y-1):] = float(angularVelo_mul[int(amp)])
                true_Ap = get_A(angularVelo_mul[amp]) 
                odetruep = TrueFunc(true_Ap)
                true_xp = odeint(odetruep, true_x[-1], tp) # run ode solver for all tp when tp[0] is at true_x[-1]
            # switched_seq_idx = generator.integers(0,self.num_seq)
            # window_img[self.y_window_len : self.sample_len] = self.all_Y[switched_seq_idx, endseq:endseq_p]
            # window_lbl[self.y_window_len - 1 : self.sample_len] = self.all_X[switched_seq_idx, endseq-1:endseq_p]
            else:
                true_xp = odeint(odetrue, true_x[-1], tp)
            
        # print(true_xp.size())
            for i in range(int(precompute_len_y)):
                dset[split_name]['signal'][idx, i,int(idx_sam)]=true_x[i+1,1]
                dset[split_name]['label'][idx,i,int(idx_sam)]=window_lbl[i]
            assert true_x[-1,1] == true_xp[0,1] and true_x[-1,0] == true_xp[0,0]
            for i in range(int(precompute_len_y), int(precompute_len_y+precompute_len_yp)):
                dset[split_name]['signal'][idx,i,int(idx_sam)]=true_xp[i-precompute_len_y+1,1]
                dset[split_name]['label'][idx,i,int(idx_sam)]=window_lbl[i]
            
            cnt_tot+=1
            idx_sam += 1
    print(float(cnt_swt)/float(cnt_tot), cnt_tot)

print('dataset size')
print('train', dset['train']['signal'].shape, dset['train']['label'].shape)
print('valid', dset['valid']['signal'].shape, dset['valid']['label'].shape)
print('test', dset['test']['signal'].shape, dset['test']['label'].shape)


train_dump_dir = dump_dir + 'train'
if not os.path.exists(train_dump_dir):
    os.makedirs(train_dump_dir)
save_npy(os.path.join(train_dump_dir, 'signals_data.npy'), dset['train']['signal'])
save_npy(os.path.join(train_dump_dir, 'freq_class.npy'), dset['train']['label'])

val_dump_dir = dump_dir + 'valid'
if not os.path.exists(val_dump_dir):
    os.makedirs(val_dump_dir)
save_npy(os.path.join(val_dump_dir, 'signals_data.npy'), dset['valid']['signal'])
save_npy(os.path.join(val_dump_dir, 'freq_class.npy'), dset['valid']['label'])

test_dump_dir = dump_dir + 'test'
if not os.path.exists(test_dump_dir):
    os.makedirs(test_dump_dir)
save_npy(os.path.join(test_dump_dir, 'signals_data.npy'), dset['test']['signal'])
save_npy(os.path.join(test_dump_dir, 'freq_class.npy'), dset['test']['label'])

print('finished generating, now plotting visualizations for you')

# visualize testing set
for signal, label in zip(dset['valid']['signal'], dset['valid']['label']):
    print(label)
    fig=plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)
    ax1.plot(np.arange(1,precompute_len_y+1), signal[:(precompute_len_y), 0], 'r.')
    ax1.plot(np.arange(precompute_len_y+1, precompute_len_y+precompute_len_yp+1), signal[(precompute_len_y):(precompute_len_y+precompute_len_yp), 0], 'b.')
    ax2.plot(np.arange(1,precompute_len_y+1), signal[:(precompute_len_y), 1], 'r.')
    ax2.plot(np.arange(precompute_len_y+1, precompute_len_y+precompute_len_yp+1), signal[(precompute_len_y):(precompute_len_y+precompute_len_yp), 1], 'b.')
    ax3.plot(np.arange(1,precompute_len_y+1), signal[:(precompute_len_y), 2], 'r.')
    ax3.plot(np.arange(precompute_len_y+1, precompute_len_y+precompute_len_yp+1), signal[(precompute_len_y):(precompute_len_y+precompute_len_yp), 2], 'b.')
    ax4.plot(np.arange(1,precompute_len_y+1), signal[:(precompute_len_y), 3], 'r.')
    ax4.plot(np.arange(precompute_len_y+1, precompute_len_y+precompute_len_yp+1), signal[(precompute_len_y):(precompute_len_y+precompute_len_yp), 3], 'b.')
    ax5.plot(np.arange(1,precompute_len_y+1), signal[:(precompute_len_y), 4], 'r.')
    ax5.plot(np.arange(precompute_len_y+1, precompute_len_y+precompute_len_yp+1), signal[(precompute_len_y):(precompute_len_y+precompute_len_yp), 4], 'b.')
    plt.show()
    input()    
