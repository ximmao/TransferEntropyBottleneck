from pickle import TRUE
import sys
import os
import math
import typing
import logging
import time
from pathlib import Path
import warnings
import argparse
import pdb
import glob
import re

import numpy as np
import torch
import torch.nn as nn
from torch.nn import parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['axes.xmargin'] = 0
import copy

sys.path.append('../')
from utils import print_network, get_plotting_func, onehot
from utils import mseloss_to_loglikelyhood, TargetLoss, r2_loss
from utils import str2bool
from models import VAEModel, VAEModel_BE, CVAEModel, TEModel, ColorClassifier
from models.misc import init_weight
from yaml_config import getStructuredArgs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main_TE(args, train_loader=None, val_loader=None):
    
    #######################################################
    # build Y_module
    
    Y_module = args.Y_module_type(**args.Y_module_args_dict)

    if args.parallel:
        Y_module = nn.DataParallel(Y_module)
    Y_module = Y_module.to(args.device)

    Y_module.apply(init_weight)

    if args.Y_checkpoint > 0:
        if args.Y_checkpoint_foldername:
            model_path = f'{args.log_dir}/{args.Y_checkpoint_foldername}/Y_module_{args.Y_checkpoint}.ckpt'
        else:
            model_path = f'{args.log_dir}/{args.exp_name}/Y_module_{args.Y_checkpoint}.ckpt'
        Y_module.load_state_dict(torch.load(model_path,map_location=args.device))
        print(f'Resume from Y checkpoint epoch {args.Y_checkpoint}')
        Y_module_at_train = args.Y_continuetrain
    else:
        print('Train Y module from scratch')    
        Y_module_at_train = True

    ###########################################################
    # build TE_module

    model = TEModel(Y_module=Y_module, **args.TE_module_args_dict)

    if args.parallel:
        model = nn.DataParallel(model)
    model = model.to(args.device)

    params = model.parameters()
    print_network(model)

    if args.TE_checkpoint > 0:
        if args.TE_checkpoint_foldername:
            model_path = f'{args.log_dir}/{args.TE_checkpoint_foldername}/TE_{args.TE_checkpoint}.ckpt'
            print(f'Resume from other TE checkpoint folder {args.TE_checkpoint_foldername}')
        else:
            model_path = f'{args.log_dir}/{args.exp_name}/TE_{args.TE_checkpoint}.ckpt'
        model.load_state_dict(torch.load(model_path))
        print(f'Resume from TE checkpoint epoch {args.TE_checkpoint}')
        print(model_path)
    else:
        print('Train TE model from scratch')
    
    color_clf = None
    if args.color_clf:
        color_clf = ColorClassifier(3, 32, 7).to(args.device)
        color_clf.load_state_dict(torch.load(args.color_clf_ckpt))

    
    print('Create train & valid datasets')
    train_dset = args.dataset_class(**args.trainset_argu)
    valid_dset = args.dataset_class(**args.validset_argu)  # NOTE: this is validation data
    train_loader = DataLoader(train_dset, batch_size=args.test_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dset, batch_size=args.test_batch_size, shuffle=False)
    
    print('Create test dataset')
    test_dset = args.dataset_class(**args.testset_argu)
    test_loader = DataLoader(test_dset, batch_size=args.test_batch_size, shuffle=False)
    
    # create plot function
    plot_results = get_plotting_func(args.dataset_name)

    
    #return_test = test_TE(args, train_loader, model, plot_results, color_clf, split='train')
    #return_test = test_TE(args, valid_loader, model, plot_results, color_clf, split='valid')
    return_test = test_TE(args, test_loader, model, plot_results, color_clf, split='test')
    if args.color_clf:
        test_loss_all, test_loss_recon, test_loss_recon_loglikelihood, test_loss_kldiv, test_loss_y_recon, test_loss_y_recon_loglikelihood, test_loss_y_kldiv, test_corrected = return_test
        test_log = (test_loss_all, test_loss_recon, test_loss_recon_loglikelihood, test_loss_kldiv, test_loss_y_recon, test_loss_y_recon_loglikelihood, test_loss_y_kldiv, test_corrected)
    test_loss = np.mean(test_loss_all)
    return model, test_loss, test_log

def test_TE(args, data_loader, model, plot_results=None, color_clf=None, split='test'):
    model.eval()
    loss_all = []
    loss_recon = []
    loss_recon_loglikelihood = []
    loss_kldiv = []
    loss_y_recon = []
    loss_y_recon_loglikelihood = []
    loss_y_kldiv = []
    metric_I_z_yout_given_c = []
    metric_I_z_x_given_c = []
    output_metric_I_z_x_given_c = []
    if args.color_clf:
        color_accuracy = []
    print('evaluate on split', split) 
    restack = True
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            
            images_trgt, images_hist, labels_trgt = batch

            images_trgt = images_trgt.to(device=args.device,dtype=torch.float)
            images_hist = images_hist.to(device=args.device,dtype=torch.float)
            if args.dataset_name == 'ColoredBouncingBallsStackedOnlinegen':
                labels_trgt = labels_trgt.to(device=args.device,dtype=torch.float) #B,seq + seq_prediction,c,h,w
            # elif args.dataset_name == 'FrequencyChangingSinesOnlinegen' or args.dataset_name == 'FrequencyChangingSinesSummedMultiple':
            elif args.dataset_name == 'FrequencyChangingSinesSummedMultiple':
                labels_trgt = labels_trgt.to(device=args.device,dtype=torch.float) #B,seq + seq_prediction,1
            else:
                labels_trgt = labels_trgt.to(device=args.device,dtype=torch.long)
            labels_hist = labels_trgt[:,:images_hist.shape[1]]
            labels_trgt = labels_trgt[:,[images_hist.shape[1]-1]]

            model.zero_grad()
            
            if args.output_categorical:
                pred, kl_div, _, y_pred, y_kl_div, I_z_x_given_c = model(images_trgt[:,0], labels_hist[:,0], images_trgt, y_next_label = labels_trgt, stopgradient=args.y_stopgradient,deterministic=args.deterministic_baseline)
            else:
                pred, kl_div, _, y_pred, y_kl_div, I_z_x_given_c = model(labels_hist, images_hist, images_trgt, y_next_label = labels_trgt, stopgradient=args.y_stopgradient,deterministic=args.deterministic_baseline)        
            
            if args.output_categorical:
                reconstructionloss,loglikelyhood = args.criterion(pred,labels_trgt[:,0])
            elif args.output_seq_scalar:
                reconstructionloss,loglikelyhood = args.criterion(pred,images_trgt)
            else:
                reconstructionloss,loglikelyhood = args.criterion(pred,images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
            
            
            if not args.true_latent_loss:
                raise ValueError()  
            else:
                loss =  args.kappa*reconstructionloss + I_z_x_given_c # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)  

            if args.Y_continuetrain:
                if args.output_categorical:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,labels_trgt[:,0])
                elif args.output_seq_scalar:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,images_trgt)
                else:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred, images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                
                y_loss =  args.kappa_Y*y_reconstructionloss + y_kl_div # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)
                
                loss = loss + y_loss

            #information metrics
            if not args.Y_continuetrain:
                if args.output_categorical:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,labels_trgt[:,0])
                elif args.output_seq_scalar:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,images_trgt)
                else:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred, images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                
            
            I_z_yout_given_c, output_I_z_x_given_c = model.info_metrics(reconstruction_loglikelihood=loglikelyhood, reconstruction_loglikelihood_y=y_loglikelyhood, kl_div=kl_div)
            
            if not (color_clf == None): 
                if args.plot_sigmoid:
                    pred_color = color_clf(torch.sigmoid(pred))
                else:
                    pred_color = color_clf(pred)
                trgt_color = color_clf(images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                _, pred_predicted = torch.max(pred_color, 1)
                _, trgt_predicted = torch.max(trgt_color, 1)
                # print('idx', batch_idx, pred_predicted, trgt_predicted)
                correct = (pred_predicted == trgt_predicted).float().detach().cpu().numpy().tolist()
                color_accuracy += correct

            loss_all.append(loss.item())
            loss_recon.append(reconstructionloss.item())
            loss_recon_loglikelihood.append(loglikelyhood.item())
            loss_kldiv.append(kl_div.item())
            loss_y_recon.append(y_reconstructionloss.item())
            loss_y_recon_loglikelihood.append(y_loglikelyhood.item())
            loss_y_kldiv.append(y_kl_div.item())
            metric_I_z_yout_given_c.append(I_z_yout_given_c.item())
            metric_I_z_x_given_c.append(I_z_x_given_c.item())
            output_metric_I_z_x_given_c.append(output_I_z_x_given_c.item())
            
            # plot
            if ((batch_idx % 1) == 0) or (batch_idx == (len(data_loader)-1)):
                plot_results(pred.detach().cpu().numpy()[0], 
                            images_trgt.detach().cpu().numpy()[0], 
                            images_hist.detach().cpu().numpy()[0], 
                            labels_hist.detach().cpu().numpy()[0], f'{batch_idx}', path=f'{args.log_dir}/{args.exp_name}_{args.seed}_forPlotting/run_plots/{split}_plots/',save=args.savefig,sigmoid=args.plot_sigmoid, plot_shifted_pixel=True)
                
            if restack:
                pred_stack = copy.deepcopy(pred.detach().cpu().numpy())
                images_trgt_stack = copy.deepcopy(images_trgt.detach().cpu().numpy())
                images_hist_stack = copy.deepcopy(images_hist.detach().cpu().numpy())
                labels_hist_stack = copy.deepcopy(labels_hist.detach().cpu().numpy())
                restack = False
            else:
                pred_stack = np.concatenate((pred_stack, pred.detach().cpu().numpy()), 0)
                images_trgt_stack = np.concatenate((images_trgt_stack, images_trgt.detach().cpu().numpy()), 0)
                images_hist_stack = np.concatenate((images_hist_stack, images_hist.detach().cpu().numpy()), 0)
                labels_hist_stack = np.concatenate((labels_hist_stack, labels_hist.detach().cpu().numpy()), 0)
            del loss, pred, kl_div, _, y_pred, y_kl_div, reconstructionloss,loglikelyhood, y_reconstructionloss,y_loglikelyhood
            
            # congregate multiple samples to show a big images
            if ((batch_idx % 10) == 9) or (batch_idx == (len(data_loader)-1)):
                plot_results(pred_stack, images_trgt_stack, images_hist_stack, 
                         labels_hist_stack, f'{batch_idx}', path=f'{args.log_dir}/{args.exp_name}_{args.seed}_forPlotting/run_plots/{split}_plots/',save=args.savefig,sigmoid=args.plot_sigmoid, plot_shifted_pixel=True, has_batch_first=True)
                restack = True
                pred_stack = None
                images_trgt_stack = None
                images_hist_stack = None
                images_hist_stack = None
        if args.save:
            torch.save((loss_all,loss_recon,loss_recon_loglikelihood,loss_kldiv,loss_y_recon,loss_y_recon_loglikelihood,loss_y_kldiv,metric_I_z_yout_given_c,metric_I_z_x_given_c,output_metric_I_z_x_given_c),f'{args.log_dir}/{args.exp_name}_{args.seed}_forPlotting/TE_module_Info_{split}_stats.pkl')

        print(f'{split} loss', f'loss is: {np.mean(loss_all)}',f'reconstructionloss is: {np.mean(loss_recon)}','recon_loglikelihood', np.mean(loss_recon_loglikelihood), 'kl', np.mean(loss_kldiv))
        print(f'Y_reconstructionloss is: {np.mean(loss_y_recon)}','Y_recon_loglikelihood', np.mean(loss_y_recon_loglikelihood), 'Y_kl', np.mean(loss_y_kldiv))
        print(f'metrics are: I(Z,Y_prime|C) {I_z_yout_given_c}, and I(X,Z|C) {I_z_x_given_c}; calculated at the output I(X,Z|C) is {output_I_z_x_given_c}')
        if args.color_clf:
            print(f'color accuracy is {100.*np.mean(color_accuracy)}')        

    if args.color_clf:
        return loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv, loss_y_recon, loss_y_recon_loglikelihood, loss_y_kldiv, color_accuracy
    else:
        return loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv, loss_y_recon, loss_y_recon_loglikelihood, loss_y_kldiv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    projectdir = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument(
        '--projectdir',
        type=str,
        default=projectdir,
        help='directory to this project')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(projectdir,'run_outputs'),
        help='directory of the log')
    parser.add_argument(
        '--config_file',
        type=str,
        default='arguments.yaml',
        help='name of config file yaml')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        metavar='seed',
        help='random seed, default 0')
    parser.add_argument(
        '--device',
        type=str,
        default=device)
    parser.add_argument(
        '--save',
        type=str2bool,
        default=True,
        help='save checkpoints or not')
    parser.add_argument(
        '--savefig',
        type=str2bool,
        default=True,
        help='save plots or not')
    parser.add_argument(
        '--parallel',
        type=str2bool,
        default=False,
        help='enable model in parallel or not')
    parser.add_argument(
        '--y_stopgradient',
        type=str2bool,
        default=True,
        help='set stopgradient to true to stop gradient flow to Y_model through the latent state c')
    parser.add_argument(
        '--Y_continuetrain',
        type=str2bool,
        default=False,
        help = 'continue to train the Y_model (with its usual objetive) when training the full TE model')
    parser.add_argument(
        '--Y_checkpoint',
        type=int,
        default=-1,
        help = 'epoch to load the Y module and optimizer from, and start at this epoch if training Y')
    parser.add_argument(
        '--Y_checkpoint_foldername',
        type=str,
        default='',
        help = 'folder to load Y module from, if empty string then uses same folder as TE')
    parser.add_argument(
        '--TE_checkpoint',
        type=int,
        default=-1,
        help = 'epoch to load the full TE module and optimizer from, and start at this epoch if training TE')
    parser.add_argument(
        '--TE_checkpoint_foldername',
        type=str,
        default='',
        help = 'folder to load the full TE module and optimizer from, and start at this epoch if training TE')
    
    ap_args = parser.parse_args()

    args = getStructuredArgs(f'./{ap_args.config_file}', ap_args)
    
    if args.output_categorical:
        args.criterion = TargetLoss(output_type = 'categorical',domain_shape=args.TE_module_args_dict['X_module_args_dict']['input_dim'],presumed_variance=args.presumed_output_variance)
    elif args.output_seq_scalar:
        args.criterion = TargetLoss(output_type = args.loss_type,domain_shape=args.signal_shape,presumed_variance=args.presumed_output_variance)
    else:
        args.criterion = TargetLoss(output_type = args.loss_type,domain_shape=args.image_shape,presumed_variance=args.presumed_output_variance)
    if args.loss_type == 'binary':
        args.plot_sigmoid = True
    else:
        args.plot_sigmoid = False
    
    if args.output_categorical:
        args.normalizing_factor_loglikelihood = 1
    elif args.output_seq_scalar:
        args.normalizing_factor_loglikelihood = np.prod(args.signal_shape)
    else:
        args.normalizing_factor_loglikelihood = np.prod(args.image_shape)
    args.kappa_Y = args.normalizing_factor_loglikelihood*args.beta_Y
    args.kappa = args.normalizing_factor_loglikelihood*args.beta_TE
    
    args.exp_name = str(args.exp_name)+'_'+str(args.seed)

    # print arguments
    for arg_name in vars(args):
        print(arg_name, ': ', getattr(args, arg_name))


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.color_clf:
        if 'g1_3' in args.testset_argu['directory']:
            args.color_clf_ckpt='ColorCLF.ckpt'
        else:
            raise ValueError('dataset folder is expected to have g1_3 in it')
        print(f'Load color classifier from {args.color_clf_ckpt}')

    args_path = f'{args.log_dir}/{args.exp_name}/args.ckpt'

    seed_list = [1]
    total_corrected = []
    assert args.TE_checkpoint > -1.
    for test_seed in seed_list:
        args.seed = test_seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print('test seed', args.seed)
        TE_model, test_loss, test_log = main_TE(args)
        total_corrected += test_log[-1]
        print(f'Testing loss on TE model is {test_loss}')
    print('accuracy across seed list', 100.*np.mean(total_corrected))