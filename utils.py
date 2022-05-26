import os, sys
import math
from typing import Type, Any, Callable, Union, List, Optional
from typing_extensions import Literal
import torch
import torch.nn as nn

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
import copy


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def onehot(x,num_classes):
    out=torch.eye(num_classes).to(x.device)[x.long()]
    return out

def build_grid(resolution): #returns shape (1,resolution[0],resolution[1],4)
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)

def mseloss_to_loglikelyhood(mse,outputsize,variance=1):
    # converts mse to loglikelyhood for gaussian model with mean equal to the activation of the model, and a fixed variance
    # return (-1/2)*(mse*outputsize/variance + math.log(variance**(outputsize)) + outputsize*math.log(2*math.pi))
    return (-1/2)*outputsize*(mse/variance + math.log(variance*2*math.pi)) # <log(d(y'|z,c))>

def l1loss_to_loglikelyhood(l1,outputsize,variance=1):
    # converts l1 to loglikelyhood for laplace model with mean equal to the activation of the model, and a fixed variance
    b = math.sqrt(variance/2)
    # return (-1.)*(l1*outputsize/b + math.log(b**(outputsize)) + outputsize*math.log(2))
    return (-1.)*outputsize*(l1/b + math.log(b*2)) # <log(d(y'|z,c))>

class MSEtoLoglikelyhood(nn.Module):
    """
    module form of mseloss_to_loglikelyhood
    """

    def __init__(self,outputresolution,variance=1,loss=False):
        super().__init__()
        self.outputresolution = outputresolution
        self.outputsize = np.prod(outputresolution)
        self.variance = variance
        if loss:
            self.multiplier = -1.0/self.outputsize
        else:
            self.multiplier = 1.0
    
    def forward(self,mse,_target=None):
        return self.multiplier*mseloss_to_loglikelyhood(mse,self.outputsize,self.variance)

class L1toLoglikelyhood(nn.Module):
    """
    module form of l1loss_to_loglikelyhood
    """

    def __init__(self,outputresolution,variance=1,loss=False):
        super().__init__()
        self.outputresolution = outputresolution
        self.outputsize = np.prod(outputresolution)
        self.variance = variance
        if loss:
            self.multiplier = -1.0/self.outputsize
        else:
            self.multiplier = 1.0
    
    def forward(self,l1,_target=None):
        return self.multiplier*l1loss_to_loglikelyhood(l1,self.outputsize,self.variance)

class TargetLoss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', 
                output_type: Literal['mse','binary','gaussian','laplace','l1']=['gaussian'],
                 presumed_variance = 1, domain_shape = (32,32)):
        super().__init__()
        self.output_type = output_type
        self.domain_shape = domain_shape
        self.num_pixels = np.prod(domain_shape)
        self.presumed_variance = presumed_variance
        self.output_type = output_type

        if self.output_type == 'mse':
            self.transform=nn.Identity()
            self.loss_fn = nn.MSELoss(size_average, reduce, reduction)
        elif self.output_type == 'binary':
            self.transform=nn.Identity() # logits as inputs
            self.loss_fn = nn.BCEWithLogitsLoss(size_average, reduce, reduction)
        elif self.output_type == 'gaussian':
            self.transform=nn.Identity()
            self.loss_fn = MultiSequential(nn.MSELoss(size_average, reduce, reduction),MSEtoLoglikelyhood(domain_shape,presumed_variance,loss=True))
        elif self.output_type == 'laplace':
            self.transform=nn.Identity()
            self.loss_fn = MultiSequential(nn.L1Loss(size_average, reduce, reduction),L1toLoglikelyhood(domain_shape,presumed_variance,loss=True))
        elif self.output_type == 'l1':
            self.transform=nn.Identity()
            self.loss_fn = nn.L1Loss(size_average, reduce, reduction)
        elif self.output_type == 'categorical':
            self.transform=nn.Identity()
            self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise 'not implemented'


    def forward(self,output,target):

        target = self.transform(target)

        if self.output_type == 'categorical':
            loss = self.loss_fn(output.clone(),target.clone())
            loglikelihood = -loss
        elif target.shape[1] == 1 or target.shape[1] == 3: # for mnist grey or bb rgb
            loss = self.loss_fn(output.clone(),target.clone())

            #loglikelihood computation
            if self.output_type == 'mse':
                loglikelihood = mseloss_to_loglikelyhood(loss,self.num_pixels,self.presumed_variance)
            elif self.output_type == 'l1':
                loglikelihood = l1loss_to_loglikelyhood(loss,self.num_pixels,self.presumed_variance)
            else:
                loglikelihood = -self.num_pixels*loss
        else:
            raise

        return loss, loglikelihood

def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))

# plotting factory 
def get_plotting_func(dataset_name='VariedRotatingDigits'):
    if 'VariedRotatingDigits' in dataset_name:
        return partial(plot_results, is_2d=True, is_grey=True, img_label=False)
    elif 'ColoredBouncingBallsStacked' in dataset_name:
        return partial(plot_results, is_2d=True, is_grey=False, img_label=True, is_stack=True)
    else:
        raise NotImplementedError()

def plot_results(pred, trgt, input_y, input_x, idx, path, is_2d, is_grey, img_label, is_stack=False, has_batch_first=False, save=True, sigmoid=False, plot_true_pixel=False, plot_shifted_pixel=False):
    """
    plot results for different datasets
      take everything normalized as numpy
      is_2d: is output 2d or 1d
      is_grey: plot greylevel or cmap
      img_label: with image as label? normal vs needle datasets 
    """
    if sigmoid:
        pred = 1./(1. + np.exp(-pred))
    pred = np.clip(pred,0,1) # for mnist exp
    if has_batch_first:
        if is_2d and (not is_grey) and is_stack and img_label and plot_shifted_pixel: # needle
            import matplotlib as mpl
            input_x_copy = copy.deepcopy(input_x)
            input_y_copy = copy.deepcopy(input_y)
            
            params={'figure.figsize':(7,11), 'axes.titlesize':8, 'axes.titleweight':'bold'}
            mpl.rcParams.update(params)
            fig=plt.figure()
            ax1 = plt.subplot2grid((1, 5), (0, 4))
            ax2 = plt.subplot2grid((1, 5), (0, 3))
            ax4 = plt.subplot2grid((1, 5), (0, 0), colspan=3)
            ax1.axis('off')
            ax2.axis('off')
            ax4.axis('off')
            ax2.set_title('Target')
            ax1.set_title('Predicted')
            ax4.set_title('Input X (transformed with shifted pixels)')
            pred = spread_seq_img(np.expand_dims(pred, 1), has_batch_first)
            trgt = spread_seq_img(trgt, has_batch_first)
            ax1.imshow(np.transpose(np.squeeze(pred*1.),(1,2,0)), vmin=0., vmax=1.)
            ax2.imshow(np.transpose(np.squeeze(trgt*1.),(1,2,0)), vmin=0., vmax=1.)
            per_sample_input_x = collapse_pixel_needle(input_x_copy[0], input_y_copy[0], False)
            per_sample_input_x = per_sample_input_x[np.newaxis, :]
            for b_idx in range(1, input_x_copy.shape[0]):
                per_sample_input_x_tmp = collapse_pixel_needle(input_x_copy[b_idx], input_y_copy[b_idx], False)
                per_sample_input_x_tmp = per_sample_input_x_tmp[np.newaxis, :]
                per_sample_input_x = np.concatenate((per_sample_input_x, per_sample_input_x_tmp), 0)
            exp_input_x = spread_seq_img(per_sample_input_x, has_batch_first)
            ax4.imshow(np.transpose(exp_input_x*1.,(1,2,0)), vmin=0., vmax=1.)
        elif is_2d and is_grey and (not img_label): # mnist
            import matplotlib as mpl
            input_y_copy = copy.deepcopy(input_y)
            
            params={'figure.figsize':(7,11), 'axes.titlesize':8, 'axes.titleweight':'bold'}
            mpl.rcParams.update(params)
            fig=plt.figure()
            ax1 = plt.subplot2grid((1, 5), (0, 3))
            ax2 = plt.subplot2grid((1, 5), (0, 2))
            ax4 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
            ax1.axis('off')
            ax2.axis('off')
            ax4.axis('off')
            ax2.set_title('Target')
            ax1.set_title('Predicted')
            ax4.set_title('Input Y')
            pred = spread_seq_img(np.expand_dims(pred, 1), has_batch_first)
            trgt = spread_seq_img(trgt, has_batch_first)
            ax1.imshow(np.squeeze(pred*255), cmap='gray', vmin=0, vmax=255)
            ax2.imshow(np.squeeze(trgt*255), cmap='gray', vmin=0, vmax=255)
            per_sample_input_y = input_y_copy[[0]]
            for b_idx in range(1, input_y_copy.shape[0]):
                per_sample_input_y_tmp = input_y_copy[[b_idx]]
                per_sample_input_y = np.concatenate((per_sample_input_y, per_sample_input_y_tmp), 0)
            exp_input_y = spread_seq_img(per_sample_input_y, has_batch_first)
            ax4.imshow(np.squeeze(exp_input_y*255), cmap='gray', vmin=0, vmax=255)
        else:
            raise NotImplementedError()
        plt.tight_layout()
        plt.axis('off')

        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, f'batch_plot_{idx}.png'))
        plt.clf()
        plt.close()
        return 

    if is_stack:
        if img_label:
            if plot_shifted_pixel:
                input_x_copy = copy.deepcopy(input_x)
                input_y_copy = copy.deepcopy(input_y)
            if plot_true_pixel:
                k = input_y.shape[1]//3
                for m in range(input_y.shape[1]//3):
                    if input_y[0,int(m*3), 1, 1] == input_x[0, int(m*3), 1, 1]:
                        pass
                    else:
                        k = m
                        break
                input_x = input_x[:,int(k*3):int((k+1)*3)]
            else:
                input_x = input_x[:, :3]
        input_y = input_y[:, :3]

    if img_label:
        fig=plt.figure(figsize=(8, 3))
        if has_batch_first:
            ax1 = fig.add_subplot(133)        
            ax2 = fig.add_subplot(132)
            ax4 = fig.add_subplot(131)
            ax1.axis('off')
            ax2.axis('off')
            ax4.axis('off')
        else:
            ax3 = fig.add_subplot(221)
            ax4 = fig.add_subplot(223)
            ax1 = fig.add_subplot(222)        
            ax2 = fig.add_subplot(224)       
            ax2.set_title('Target')
            ax1.set_title('Predicted')
            ax3.set_title('Input_Y (first 3 channels)')
            if plot_true_pixel:
                ax4.set_title('Input_X (true pixel channels)')
            elif plot_shifted_pixel:
                ax4.set_title('Input_X (transformed with shifted pixels)') 
            else:
                ax4.set_title('Input_X (first 3 channels)')
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')
    else:
        fig=plt.figure(figsize=(5,6))
        ax3 = fig.add_subplot(212)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax2.set_title('Target')
        ax1.set_title('Predicted')
        ax3.set_title('Input_Y')
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')

    if is_2d:
        shape_len = len(pred.shape)
        if shape_len == 5:
            assert has_batch_first
            pass # TODO
        elif shape_len == 3:
            assert not has_batch_first
            if is_grey:
                ax1.imshow(np.squeeze(pred*255), cmap='gray', vmin=0, vmax=255)
                ax2.imshow(np.squeeze(trgt*255), cmap='gray', vmin=0, vmax=255)
            else:
                ax1.imshow(np.transpose(pred*1.,(1,2,0)), vmin=0., vmax=1.)
                ax2.imshow(np.transpose(np.squeeze(trgt*1.),(1,2,0)), vmin=0., vmax=1.)
        elif shape_len == 4:
            if has_batch_first:
                pass # TODO
            else:
                exp_pred = spread_seq_img(pred, has_batch_first)
                exp_trgt = spread_seq_img(trgt, has_batch_first)
                if is_grey:
                    ax1.imshow(np.squeeze(exp_pred*255),cmap='gray', vmin=0, vmax=255)
                    ax2.imshow(np.squeeze(exp_trgt*255),cmap='gray', vmin=0, vmax=255)
                else:
                    ax1.imshow(np.transpose(exp_pred*1.,(1,2,0)), vmin=0., vmax=1.)
                    ax2.imshow(np.transpose(exp_trgt*1.,(1,2,0)), vmin=0., vmax=1.)
        
        shape_len = len(input_y.shape)
        assert shape_len in [4, 5]
        if shape_len == 5:
            assert has_batch_first
            pass # TODO
        elif shape_len == 4:
            exp_input_y = spread_seq_img(input_y, has_batch_first)
            if is_grey:
                ax3.imshow(np.squeeze(exp_input_y*255),cmap='gray', vmin=0, vmax=255)
            else:
                ax3.imshow(np.transpose(exp_input_y*1.,(1,2,0)), vmin=0., vmax=1.)

        if img_label:
            shape_len = len(input_x.shape)
            assert shape_len in [4, 5] # assume seq as input_y
            if shape_len == 5:
                assert has_batch_first
                pass # TODO
            elif shape_len == 4:
                if plot_shifted_pixel:
                    exp_input_x = collapse_pixel_needle(input_x_copy, input_y_copy, has_batch_first)
                    exp_input_x = spread_seq_img(exp_input_x, has_batch_first)
                else:
                    exp_input_x = spread_seq_img(input_x, has_batch_first)
                if is_grey:
                    ax4.imshow(np.squeeze(exp_input_x*255),cmap='gray', vmin=0, vmax=255)
                else:
                    ax4.imshow(np.transpose(exp_input_x*1.,(1,2,0)), vmin=0., vmax=1.)
    else:
        raise NotImplementedError()
    plt.tight_layout()
    plt.axis('off')

    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f'epoch{idx}.png'))
    plt.clf()
    plt.close()

def spread_seq_img(seq_img, has_batch_first=False):
    # assert seq_img is (batch if has_batch_first) x seq x img_size
    if has_batch_first:
        seq_dim = int(1)
        bs_dim = int(0)
        
        expanded_img = seq_img[0,0]
        for m in range(1, seq_img.shape[seq_dim]):
            expanded_img = np.concatenate((expanded_img, seq_img[0, m]), axis=-1)
        
        for n in range(1, seq_img.shape[bs_dim]):
            expanded_img_row = seq_img[n,0]
            for m in range(1, seq_img.shape[seq_dim]):
                expanded_img_row = np.concatenate((expanded_img_row, seq_img[n, m]), axis=-1)
            expanded_img = np.concatenate((expanded_img, expanded_img_row), axis=-2)
        
    else:
        seq_dim = int(0)
        expanded_img = seq_img[0]
        for m in range(1, seq_img.shape[seq_dim]):
            expanded_img = np.concatenate((expanded_img, seq_img[m]), axis=-1)
    
    return expanded_img

def collapse_pixel_needle(input_x, input_y, has_batch_first=False):
    if has_batch_first:
        raise NotImplementedError()
    else:
        k = input_y.shape[1]//3
        for m in range(input_y.shape[1]//3):
            if (input_y[0,int(m*3), 1, 1] == input_x[0, int(m*3), 1, 1]) and (input_y[0,int(m*3+1), 1, 1] == input_x[0, int(m*3+1), 1, 1]) and (input_y[0,int(m*3+2), 1, 1] == input_x[0, int(m*3+2), 1, 1]):
                pass
            else:
                k = m
                break
        exp_input_x = copy.deepcopy(input_x[:,int(k*3):int((k+1)*3)])
        potential_loc = [[1, 13], [27, 30], [30, 15], [1, 7], [30, 1], [6, 30], [30, 9], [2, 30], [15, 30], [1, 20], 
                         [7, 1], [30, 13], [1,10],[30, 30], [11, 30], [29, 30], [30, 26], [28, 1], [30, 19], [20, 1]]
        for m in range(int(min(10, input_y.shape[1]//3))):
            dis_loc = potential_loc[m]
            for t in range(input_y.shape[0]): # time index
                exp_input_x[t, 0, dis_loc[0], dis_loc[1]] = input_y[t, int(m*3), 1, 1]
                exp_input_x[t, 1, dis_loc[0], dis_loc[1]] = input_y[t, int(m*3+1), 1, 1]
                exp_input_x[t, 2, dis_loc[0], dis_loc[1]] = input_y[t, int(m*3+2), 1, 1]
        return exp_input_x

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')