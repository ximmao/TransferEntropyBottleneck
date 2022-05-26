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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['axes.xmargin'] = 0
import tqdm

sys.path.append('../')
from utils import print_network, get_plotting_func, onehot
from utils import mseloss_to_loglikelyhood, TargetLoss, r2_loss
from utils import str2bool
from models import VAEModel, VAEModel_BE, CVAEModel_BE, TEModel, ColorClassifier
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
            opt_path = f'{args.log_dir}/{args.Y_checkpoint_foldername}/Y_module_opt_{args.Y_checkpoint}.ckpt'
        else:
            model_path = f'{args.log_dir}/{args.exp_name}/Y_module_{args.Y_checkpoint}.ckpt'
            opt_path = f'{args.log_dir}/{args.exp_name}/Y_module_opt_{args.Y_checkpoint}.ckpt'
        Y_module.load_state_dict(torch.load(model_path,map_location=args.device))
        print(f'Resume from Y checkpoint epoch {args.Y_checkpoint}')
        if args.Y_continuetrain or not args.y_stopgradient:
            Y_module_at_train = True
        else:
            Y_module_at_train = False
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
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    print_network(model)

    if args.TE_checkpoint > 0:
        if args.TE_checkpoint_foldername:
            model_path = f'{args.log_dir}/{args.TE_checkpoint_foldername}/TE_{args.TE_checkpoint}.ckpt'
            opt_path = f'{args.log_dir}/{args.TE_checkpoint_foldername}/TE_opt_{args.TE_checkpoint}.ckpt'
            model.load_state_dict(torch.load(model_path))
            optimizer.load_state_dict(torch.load(opt_path))
            print(f'Resume from other TE checkpoint folder {args.TE_checkpoint_foldername}')
        else:
            model_path = f'{args.log_dir}/{args.exp_name}/TE_{args.TE_checkpoint}.ckpt'
            opt_path = f'{args.log_dir}/{args.exp_name}/TE_opt_{args.TE_checkpoint}.ckpt'
            model.load_state_dict(torch.load(model_path))
            optimizer.load_state_dict(torch.load(opt_path))
        print(f'Resume from TE checkpoint epoch {args.TE_checkpoint}')
    else:
        print('Train TE model from scratch')
    X_module_at_train = True
    
    color_clf = None
    if args.color_clf:
        color_clf = ColorClassifier(3, 32, 7).to(args.device)
        color_clf.load_state_dict(torch.load(args.color_clf_ckpt))


    print('Create train & valid datasets')
    train_dset = args.dataset_class(**args.trainset_argu)
    valid_dset = args.dataset_class(**args.validset_argu)
    train_loader = DataLoader(train_dset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_dset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    
    # create plot function
    plot_results = get_plotting_func(args.dataset_name)
    
    print(time.time())
    best_val = float('inf')
    loss_all = []
    loss_recon = [] 
    loss_recon_loglikelihood = [] 
    loss_kldiv = []
    val_loss_all = []
    val_loss_recon = [] 
    val_loss_recon_loglikelihood = [] 
    val_loss_kldiv = []
    for epoch in range(args.TE_checkpoint+1,args.TE_epochs):

        if args.annealbeta is not False and args.annealbeta != 'none':
            current_kappa = args.kappa
            if isinstance(args.annealbeta, dict):
                if epoch in args.annealbeta.keys():
                    newbeta = args.annealbeta[epoch]
                else:
                    newbeta = current_kappa/args.normalizing_factor_loglikelihood
            elif epoch > 4*args.TE_epochs//5:
                newbeta = args.beta_TE
            else:
                if 'exponential' in args.annealbeta:
                    newbeta = (args.stepsize**epoch)*args.startbeta_TE
                elif 'linear' in args.annealbeta:
                    newbeta = (args.stepsize*epoch) + args.startbeta_TE
                else:
                    newbeta = (args.stepsize**epoch)*args.startbeta_TE

            args.kappa=args.normalizing_factor_loglikelihood*newbeta
            if not args.kappa == current_kappa:
                print(f'\n on epoch {epoch}, and the annealing schedule now sets beta to {newbeta} \n')

        return_train = train_TE(args, train_loader, model, optimizer, epoch, X_module_at_train, Y_module_at_train, plot_results, color_clf)
        loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv, loss_y_recon, loss_y_recon_loglikelihood, loss_y_kldiv, metric_I_z_yout_given_c, metric_I_z_x_given_c, output_metric_I_z_x_given_c = return_train
        # path to save checkpoint
        if args.save and ((epoch % args.checkpoint_frequency == 0) or epoch == (args.TE_epochs - 1)) and epoch > 0:
            model_path = f'{args.log_dir}/{args.exp_name}/TE_{epoch}.ckpt'
            opt_path = f'{args.log_dir}/{args.exp_name}/TE_opt_{epoch}.ckpt'
            if not os.path.exists(f'{args.log_dir}/{args.exp_name}'):
                os.mkdir(f'{args.log_dir}/{args.exp_name}')
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
        if ((epoch % args.validation_frequency == 0) or (epoch == (args.TE_epochs - 1))) and epoch > 0:

            print('training loss at epoch', epoch, 'is', np.mean(loss_all), 'recon_loss', np.mean(loss_recon), 'recon_loglikelihood', np.mean(loss_recon_loglikelihood), 'kl', np.mean(loss_kldiv))
            print(f'Y_reconstructionloss is: {np.mean(loss_y_recon)}','Y_recon_loglikelihood', np.mean(loss_y_recon_loglikelihood), 'Y_kl', np.mean(loss_y_kldiv))
            print(f'metrics are: I(Z,Y_prime|C) {np.mean(metric_I_z_yout_given_c)}, and I(X,Z|C) {np.mean(metric_I_z_x_given_c)}; calculated at the output I(X,Z|C) is {np.mean(output_metric_I_z_x_given_c)}')
        
            print(time.time())
            return_valid = validate_TE(args, val_loader, model, epoch, plot_results, color_clf)
            val_loss_all, val_loss_recon, val_loss_recon_loglikelihood, val_loss_kldiv, val_loss_y_recon, val_loss_y_recon_loglikelihood, val_loss_y_kldiv = return_valid
            if val_loss_all[-1]<best_val:
                best_val = val_loss_all[-1]
                best_val_epoch = epoch
    return model, best_val_epoch, best_val, (loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv), (val_loss_all, val_loss_recon, val_loss_recon_loglikelihood, val_loss_kldiv)

def main_Y(args):
    
    #######################################################
    # build Y_module
    
    model = args.Y_module_type(**args.Y_module_args_dict)

    if args.parallel:
        model = nn.DataParallel(model)
    model = model.to(args.device)

    model.apply(init_weight)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    print_network(model)
    
    if args.Y_checkpoint > 0:
        if args.Y_checkpoint_foldername:
            model_path = f'{args.log_dir}/{args.Y_checkpoint_foldername}/Y_module_{args.Y_checkpoint}.ckpt'
            opt_path = f'{args.log_dir}/{args.Y_checkpoint_foldername}/Y_module_opt_{args.Y_checkpoint}.ckpt'
            print(f'Resume from other checkpoint folder {args.Y_checkpoint_foldername}')
        else:
            model_path = f'{args.log_dir}/{args.exp_name}/Y_module_{args.Y_checkpoint}.ckpt'
            opt_path = f'{args.log_dir}/{args.exp_name}/Y_module_opt_{args.Y_checkpoint}.ckpt'
        model.load_state_dict(torch.load(model_path,map_location=args.device))
        optimizer.load_state_dict(torch.load(opt_path,map_location=args.device))
        print(f'Resume from checkpoint epoch {args.Y_checkpoint}')
    else:
        print('Train Y module from scratch')
    
    model.train()

    color_clf = None
    if args.color_clf:
        color_clf = ColorClassifier(3, 32, 7).to(args.device)
        color_clf.load_state_dict(torch.load(args.color_clf_ckpt))

    ### load data ###
    print('Create train & valid datasets')
    train_dset = args.dataset_class(**args.trainset_argu)
    valid_dset = args.dataset_class(**args.validset_argu)
    train_loader = DataLoader(train_dset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_dset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    
    # create plot function
    plot_results = get_plotting_func(args.dataset_name)
    
    print(time.time())
    best_val = float('inf')
    loss_all = []
    loss_recon_mse = [] 
    loss_recon_loglikelihood = [] 
    loss_kldiv = []
    val_loss_all = []
    val_loss_recon_mse = [] 
    val_loss_recon_loglikelihood = [] 
    val_loss_kldiv = []
    train_log_y = None
    val_log_y = None
    for epoch in range(args.Y_checkpoint+1,args.Y_epochs):
        
        loss_all, loss_recon_mse, loss_recon_loglikelihood, loss_kldiv = train_Y(args, train_loader, model, optimizer, epoch, plot_results, color_clf)
        # path to save checkpoint
        if args.save and ((epoch % args.checkpoint_frequency == 0) or epoch == (args.Y_epochs - 1)) and epoch > 0:
            model_path = f'{args.log_dir}/{args.exp_name}/Y_module_{epoch}.ckpt'
            opt_path = f'{args.log_dir}/{args.exp_name}/Y_module_opt_{epoch}.ckpt'
            if not os.path.exists(f'{args.log_dir}/{args.exp_name}'):
                os.mkdir(f'{args.log_dir}/{args.exp_name}')
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
        if ((epoch % args.validation_frequency == 0) or (epoch == (args.Y_epochs - 1))) and epoch > 0:

            print('training loss at epoch', epoch, 'is', np.mean(loss_all), 'recon_loss', np.mean(loss_recon_mse), 'recon_loglikelihood', np.mean(loss_recon_loglikelihood), 'kl', np.mean(loss_kldiv))
            print(time.time())
            val_loss_all, val_loss_recon_mse, val_loss_recon_loglikelihood, val_loss_kldiv = validate_Y(args, val_loader, model, epoch, plot_results, color_clf)
            if val_loss_all[-1]<best_val:
                best_val = val_loss_all[-1]
                best_val_epoch = epoch
        train_log_y = (loss_all, loss_recon_mse, loss_recon_loglikelihood, loss_kldiv)
        val_log_y = (val_loss_all, val_loss_recon_mse, val_loss_recon_loglikelihood, val_loss_kldiv)

    return model, best_val_epoch, best_val, train_log_y, val_log_y

def train_TE(args, data_loader, model, optimizer, epoch, X_at_train=True, Y_at_train=False, plot_results=None, color_clf=None):
    model.set_train(x_to_train=X_at_train, y_to_train=Y_at_train)
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
    if args.output_categorical:
        mnist_accuracy = []
    if args.color_clf:
        color_accuracy = []
    for batch_idx, batch in enumerate(data_loader):
        
        images_trgt, images_hist, labels_trgt = batch

        images_trgt = images_trgt.to(device=args.device,dtype=torch.float) #B,seq_prediction,c,h,w
        images_hist = images_hist.to(device=args.device,dtype=torch.float) #B,seq,c,h,w
        if args.dataset_name == 'ColoredBouncingBallsStackedOnlinegen':
            labels_trgt = labels_trgt.to(device=args.device,dtype=torch.float) #B,seq + seq_prediction,c,h,w
        else:
            labels_trgt = labels_trgt.to(device=args.device,dtype=torch.long) #B,seq + seq_prediction,c,h,w
        labels_hist = labels_trgt[:,:images_hist.shape[1]]
        labels_trgt = labels_trgt[:,[images_hist.shape[1]-1]]
        model.zero_grad()

        if args.output_categorical:
            pred, kl_div, _, y_pred, y_kl_div, I_z_x_given_c = model(images_trgt[:,0], labels_hist[:,0], images_trgt, y_next_label = labels_trgt, stopgradient=args.y_stopgradient,deterministic=args.deterministic_baseline)
        else:
            pred, kl_div, _, y_pred, y_kl_div, I_z_x_given_c = model(labels_hist, images_hist, images_trgt, y_next_label = labels_trgt, stopgradient=args.y_stopgradient,deterministic=args.deterministic_baseline)
        

        if args.output_categorical:
            reconstructionloss,loglikelyhood = args.criterion(pred,labels_trgt[:,0])
        else:
            reconstructionloss,loglikelyhood = args.criterion(pred,images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
            
        if not args.true_latent_loss:
            raise ValueError()
        else:
            loss =  args.kappa*reconstructionloss + I_z_x_given_c # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)  


        if args.Y_continuetrain:
            if args.output_categorical:
                y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,labels_trgt[:,0])
            else:
                y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred, images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                
            
            y_loss =  args.kappa_Y*y_reconstructionloss + y_kl_div # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)
            
            loss = loss + y_loss
        
        loss.backward()
        optimizer.step()

        #information metrics
        with torch.no_grad():
            if not args.Y_continuetrain:
                if args.output_categorical:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,labels_trgt[:,0])
                else:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred, images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
            
            I_z_yout_given_c, output_I_z_x_given_c = model.info_metrics(reconstruction_loglikelihood=loglikelyhood, reconstruction_loglikelihood_y=y_loglikelyhood, kl_div=kl_div)

            if args.output_categorical:
                _, predicted = torch.max(pred, 1)
                mnist_correct = (predicted == labels_trgt[:,0]).float().mean().item()
        
        if ((batch_idx % 100) == 0) or (batch_idx == (len(data_loader)-1)):

            print(f'at epoch {epoch}, and batch_idx {batch_idx}',f'loss is: {loss.item()}',f'reconstructionloss is: {reconstructionloss.item()}',f'kl_div is: {kl_div.item()}')
            print(f'Y_reconstructionloss is: {y_reconstructionloss.item()}',f'Y_kl_div is: {y_kl_div.item()}')
            print(f'metrics are: I(Z,Y_prime|C) {I_z_yout_given_c}, and I(X,Z|C) {I_z_x_given_c}; calculated at the output I(X,Z|C) is {output_I_z_x_given_c}')        
            if args.output_categorical:
                print(f'mnist accuracy is {100*mnist_correct} %')        
        
        with torch.no_grad():
            if args.color_clf: 
                if args.plot_sigmoid:
                    pred_color = color_clf(torch.sigmoid(pred))
                else:
                    pred_color = color_clf(pred)
                trgt_color = color_clf(images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                _, pred_predicted = torch.max(pred_color, 1)
                _, trgt_predicted = torch.max(trgt_color, 1)
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
        if args.output_categorical:
            mnist_accuracy.append(100*mnist_correct)
        # plot
        if ((batch_idx % args.plot_per_update_train) == 0) or (batch_idx == (len(data_loader)-1)):
            plot_results(pred.detach().cpu().numpy()[0], 
                         images_trgt.detach().cpu().numpy()[0], 
                         images_hist.detach().cpu().numpy()[0], 
                         labels_hist.detach().cpu().numpy()[0], f'{epoch}_{batch_idx}', path=f'{args.log_dir}/{args.exp_name}/run_plots/train_plots/',save=args.savefig,sigmoid=args.plot_sigmoid)
            plot_results(y_pred.detach().cpu().numpy()[0], 
                         images_trgt.detach().cpu().numpy()[0], 
                         images_hist.detach().cpu().numpy()[0], 
                         labels_hist.detach().cpu().numpy()[0], f'{epoch}_{batch_idx}', path=f'{args.log_dir}/{args.exp_name}/run_plots/train_plots/Y_submodule/',save=args.savefig,sigmoid=args.plot_sigmoid)
        
        #memory is very tight, so we have to manually delete things to get ahead of garbage collection in the next batch
        # print(f'reserved {torch.cuda.memory_reserved(0)} ,allocated {torch.cuda.memory_allocated(0)}')
        del loss, pred, kl_div, _, y_pred, y_kl_div, reconstructionloss,loglikelyhood, y_reconstructionloss,y_loglikelyhood
        # print(f'reserved {torch.cuda.memory_reserved(0)} ,allocated {torch.cuda.memory_allocated(0)}')
    if args.save:
        stats_dict = {'loss_all':loss_all,'loss_recon':loss_recon,'loss_recon_loglikelihood':loss_recon_loglikelihood,'loss_kldiv':loss_kldiv,'loss_y_recon':loss_y_recon,'loss_y_recon_loglikelihood':loss_y_recon_loglikelihood,'loss_y_kldiv':loss_y_kldiv,'metric_I_z_yout_given_c':metric_I_z_yout_given_c,'metric_I_z_x_given_c':metric_I_z_x_given_c,'output_metric_I_z_x_given_c':output_metric_I_z_x_given_c}
        if args.output_categorical:
            stats_dict.update({'mnist_accuracy':mnist_accuracy})
        if args.color_clf:
            stats_dict.update({'color_accuracy':color_accuracy})
    if args.color_clf:
        print(f'train color accuracy is {100.*np.mean(color_accuracy)} %')    

    return loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv, loss_y_recon, loss_y_recon_loglikelihood, loss_y_kldiv, metric_I_z_yout_given_c, metric_I_z_x_given_c, output_metric_I_z_x_given_c

def validate_TE(args, data_loader, model, epoch, plot_results=None, color_clf=None):
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
    if args.output_categorical:
        mnist_accuracy = []
    if args.color_clf:
        color_accuracy = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            
            images_trgt, images_hist, labels_trgt = batch

            images_trgt = images_trgt.to(device=args.device,dtype=torch.float)
            images_hist = images_hist.to(device=args.device,dtype=torch.float)
            if args.dataset_name == 'ColoredBouncingBallsStackedOnlinegen':
                labels_trgt = labels_trgt.to(device=args.device,dtype=torch.float) #B,seq + seq_prediction,c,h,w
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
            else:
                reconstructionloss,loglikelyhood = args.criterion(pred,images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
            
            
            if not args.true_latent_loss:
                raise ValueError()
            else:
                loss =  args.kappa*reconstructionloss + I_z_x_given_c # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)  

            if args.Y_continuetrain:
                if args.output_categorical:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,labels_trgt[:,0])
                else:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred, images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                
                y_loss =  args.kappa_Y*y_reconstructionloss + y_kl_div # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)
                
                loss = loss + y_loss

            #information metrics
            if not args.Y_continuetrain:
                if args.output_categorical:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred,labels_trgt[:,0])
                else:
                    y_reconstructionloss,y_loglikelyhood = args.criterion(y_pred, images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                
            I_z_yout_given_c, output_I_z_x_given_c = model.info_metrics(reconstruction_loglikelihood=loglikelyhood, reconstruction_loglikelihood_y=y_loglikelyhood, kl_div=kl_div)
            
            if args.output_categorical:
                _, predicted = torch.max(pred, 1)
                mnist_correct = (predicted == labels_trgt[:,0]).float().mean().item()

            if args.color_clf: 
                if args.plot_sigmoid:
                    pred_color = color_clf(torch.sigmoid(pred))
                else:
                    pred_color = color_clf(pred)
                trgt_color = color_clf(images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                _, pred_predicted = torch.max(pred_color, 1)
                _, trgt_predicted = torch.max(trgt_color, 1)
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
            if args.output_categorical:
                mnist_accuracy.append(100*mnist_correct)            
            # plot
            if ((batch_idx % args.plot_per_update_val) == 0) or (batch_idx == (len(data_loader)-1)):
                plot_results(pred.detach().cpu().numpy()[0], 
                            images_trgt.detach().cpu().numpy()[0], 
                            images_hist.detach().cpu().numpy()[0], 
                            labels_hist.detach().cpu().numpy()[0], f'{epoch}_{batch_idx}', path=f'{args.log_dir}/{args.exp_name}/run_plots/validation_plots/',save=args.savefig,sigmoid=args.plot_sigmoid)
                plot_results(y_pred.detach().cpu().numpy()[0], 
                            images_trgt.detach().cpu().numpy()[0], 
                            images_hist.detach().cpu().numpy()[0], 
                            labels_hist.detach().cpu().numpy()[0], f'{epoch}_{batch_idx}', path=f'{args.log_dir}/{args.exp_name}/run_plots/validation_plots/Y_submodule/',save=args.savefig,sigmoid=args.plot_sigmoid)
        
            del loss, pred, kl_div, _, y_pred, y_kl_div, reconstructionloss,loglikelyhood, y_reconstructionloss,y_loglikelyhood
        if args.save:
            stats_dict = {'loss_all':loss_all,'loss_recon':loss_recon,'loss_recon_loglikelihood':loss_recon_loglikelihood,'loss_kldiv':loss_kldiv,'loss_y_recon':loss_y_recon,'loss_y_recon_loglikelihood':loss_y_recon_loglikelihood,'loss_y_kldiv':loss_y_kldiv,'metric_I_z_yout_given_c':metric_I_z_yout_given_c,'metric_I_z_x_given_c':metric_I_z_x_given_c,'output_metric_I_z_x_given_c':output_metric_I_z_x_given_c}
            if args.output_categorical:
                stats_dict.update({'mnist_accuracy':mnist_accuracy})
            if args.color_clf:
                stats_dict.update({'color_accuracy':color_accuracy})
            torch.save(stats_dict,f'{args.log_dir}/{args.exp_name}/TE_module_Info_valid_stats_{epoch}.pkl')

        print(f'validation loss at epoch {epoch}', f'loss is: {np.mean(loss_all)}',f'reconstructionloss is: {np.mean(loss_recon)}','recon_loglikelihood', np.mean(loss_recon_loglikelihood), 'kl', np.mean(loss_kldiv))
        print(f'Y_reconstructionloss is: {np.mean(loss_y_recon)}','Y_recon_loglikelihood', np.mean(loss_y_recon_loglikelihood), 'Y_kl', np.mean(loss_y_kldiv))
        print(f'metrics are: I(Z,Y_prime|C) {np.mean(metric_I_z_yout_given_c)}, and I(X,Z|C) {np.mean(metric_I_z_x_given_c)}; calculated at the output I(X,Z|C) is {np.mean(output_metric_I_z_x_given_c)}')        
        if args.output_categorical:
            print(f'mnist accuracy is {100*mnist_correct} %')   
        if args.color_clf:
            print(f'color accuracy is {100.*np.mean(color_accuracy)} %')

        print(time.time())
    return loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv, loss_y_recon, loss_y_recon_loglikelihood, loss_y_kldiv

def train_Y(args, data_loader, model, optimizer, epoch, plot_results=None, color_clf=None):
    model.train()
    loss_all = []
    loss_recon = []
    loss_recon_loglikelihood = []
    loss_kldiv = []
    if args.output_categorical:
        mnist_accuracy = []
    if args.color_clf:
        color_accuracy = []

    for batch_idx, batch in enumerate(data_loader):

        images_trgt, images_hist, labels_trgt = batch
        images_trgt = images_trgt.to(device=args.device,dtype=torch.float) #B,seq_prediction,c,h,w
        images_hist = images_hist.to(device=args.device,dtype=torch.float) #B,seq,c,h,w
        labels_trgt = labels_trgt.to(device=args.device,dtype=torch.long)
        labels_hist = labels_trgt[:,:images_hist.shape[1]]
        labels_trgt = labels_trgt[:,[images_hist.shape[1]-1]]
        model.zero_grad()

        if args.Y_module_type == VAEModel:
            _, pred, kl_div, _ = model(images_hist,deterministic=args.deterministic_baseline)
        elif args.Y_module_type == VAEModel_BE:
            if args.output_categorical:
                _, pred, kl_div, _ = model(labels_hist[:,0], labels_trgt[:,[0]],deterministic=args.deterministic_baseline)
            else:
                _, pred, kl_div, _ = model(images_hist, images_trgt,deterministic=args.deterministic_baseline)
        if args.output_categorical:
            reconstructionloss,loglikelyhood = args.criterion(pred,labels_trgt[:,0])
        else:
            reconstructionloss,loglikelyhood = args.criterion(pred,images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
        
        loss =  args.kappa_Y*reconstructionloss + kl_div # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)  
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if args.output_categorical:
                _, predicted = torch.max(pred, 1)
                mnist_correct = (predicted == labels_trgt[:,0]).float().mean().item()

            if args.color_clf: 
                if args.plot_sigmoid:
                    pred_color = color_clf(torch.sigmoid(pred))
                else:
                    pred_color = color_clf(pred)
                trgt_color = color_clf(images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                _, pred_predicted = torch.max(pred_color, 1)
                _, trgt_predicted = torch.max(trgt_color, 1)
                correct = (pred_predicted == trgt_predicted).float().detach().cpu().numpy().tolist()
                color_accuracy += correct

        
        if ((batch_idx % 100) == 0) or (batch_idx == (len(data_loader)-1)):
            print(f'at epoch {epoch}, and batch_idx {batch_idx}',f'loss is: {loss.item()}',f'reconstructionloss is: {reconstructionloss.item()}',f'kl_div is: {kl_div.item()}')
            if args.output_categorical:
                print(f'mnist accuracy is {100*mnist_correct} %')

        loss_all.append(loss.item())
        loss_recon.append(reconstructionloss.item())
        loss_recon_loglikelihood.append(loglikelyhood.item())
        loss_kldiv.append(kl_div.item())
        if args.output_categorical:
            mnist_accuracy.append(100*mnist_correct)  
        # plot
        if ((batch_idx % args.plot_per_update_train) == 0) or (batch_idx == (len(data_loader)-1)):
            plot_results(pred.detach().cpu().numpy()[0], 
                         images_trgt.detach().cpu().numpy()[0], 
                         images_hist.detach().cpu().numpy()[0], 
                         images_trgt.detach().cpu().numpy()[0], f'{epoch}_{batch_idx}', path=f'{args.log_dir}/{args.exp_name}/run_plots/Y_train_plots/',save=args.savefig,sigmoid=args.plot_sigmoid)
        
    if args.save:
        stats_dict = {'loss_all':loss_all,'loss_recon':loss_recon,'loss_recon_loglikelihood':loss_recon_loglikelihood,'loss_kldiv':loss_kldiv}
        if args.output_categorical:
            stats_dict.update({'mnist_accuracy':mnist_accuracy})
        if args.color_clf:
            stats_dict.update({'color_accuracy':color_accuracy})
        torch.save(stats_dict,f'{args.log_dir}/{args.exp_name}/Y_module_Info_train_stats_{epoch}.pkl')
    if args.color_clf:
        print(f'train color accuracy is {100.*np.mean(color_accuracy)} %')    

    return loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv

def validate_Y(args, data_loader, model, epoch, plot_results=None, color_clf=None):
    model.eval()
    loss_all = []
    loss_recon = []
    loss_recon_loglikelihood = []
    loss_kldiv = []
    if args.output_categorical:
        mnist_accuracy = []
    if args.color_clf:
        color_accuracy = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images_trgt, images_hist, labels_trgt = batch
            images_trgt = images_trgt.to(device=args.device,dtype=torch.float) #B,seq_prediction,c,h,w
            images_hist = images_hist.to(device=args.device,dtype=torch.float) #B,seq,c,h,w
            labels_trgt = labels_trgt.to(device=args.device,dtype=torch.long)
            labels_hist = labels_trgt[:,:images_hist.shape[1]]
            labels_trgt = labels_trgt[:,[images_hist.shape[1]-1]]
            model.zero_grad()

            if args.Y_module_type == VAEModel:
                _, pred, kl_div, _ = model(images_hist,deterministic=args.deterministic_baseline)
            elif args.Y_module_type == VAEModel_BE:
                if args.output_categorical:
                    _, pred, kl_div, _ = model(labels_hist[:,0], labels_trgt[:,[0]],deterministic=args.deterministic_baseline)
                else:
                    _, pred, kl_div, _ = model(images_hist, images_trgt,deterministic=args.deterministic_baseline)

            if args.output_categorical:
                reconstructionloss,loglikelyhood = args.criterion(pred,labels_trgt[:,0])
            else:
                reconstructionloss,loglikelyhood = args.criterion(pred,images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))

            loss = args.kappa_Y*reconstructionloss + kl_div # loglikelyhood = <log(d(y'|z,c))> , kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>. Rate indicates how many bits to the MNI point assuming the optimal d, and loglikelihood bounds I(Z,Y'|C)  
            
            if args.output_categorical:
                _, predicted = torch.max(pred, 1)
                mnist_correct = (predicted == labels_trgt[:,0]).float().mean().item()

            if not (color_clf == None): 
                if args.plot_sigmoid:
                    pred_color = color_clf(torch.sigmoid(pred))
                else:
                    pred_color = color_clf(pred)
                trgt_color = color_clf(images_trgt.view(images_trgt.shape[0]*images_trgt.shape[1],*images_trgt.shape[2:]))
                _, pred_predicted = torch.max(pred_color, 1)
                _, trgt_predicted = torch.max(trgt_color, 1)
                correct = (pred_predicted == trgt_predicted).float().detach().cpu().numpy().tolist()
                color_accuracy += correct

            loss_all.append(loss.item())
            loss_recon.append(reconstructionloss.item())
            loss_recon_loglikelihood.append(loglikelyhood.item())
            loss_kldiv.append(kl_div.item())
            if args.output_categorical:
                mnist_accuracy.append(100*mnist_correct) 
            # plot
            if ((batch_idx % args.plot_per_update_val) == 0) or (batch_idx == (len(data_loader)-1)):
                plot_results(pred.detach().cpu().numpy()[0], 
                             images_trgt.detach().cpu().numpy()[0], 
                             images_hist.detach().cpu().numpy()[0], 
                             images_trgt.detach().cpu().numpy()[0], f'{epoch}_{batch_idx}', path=f'{args.log_dir}/{args.exp_name}/run_plots/Y_validation_plots/',save=args.savefig,sigmoid=args.plot_sigmoid)

        if args.save:
            stats_dict = {'loss_all':loss_all,'loss_recon':loss_recon,'loss_recon_loglikelihood':loss_recon_loglikelihood,'loss_kldiv':loss_kldiv}
            if args.output_categorical:
                stats_dict.update({'mnist_accuracy':mnist_accuracy})
            if args.color_clf:
                stats_dict.update({'color_accuracy':color_accuracy})
            torch.save(stats_dict,f'{args.log_dir}/{args.exp_name}/Y_module_Info_valid_stats_{epoch}.pkl')
        print(f'validation loss at epoch {epoch}', f'loss is: {np.mean(loss_all)}',f'reconstructionloss is: {np.mean(loss_recon)}','recon_loglikelihood', np.mean(loss_recon_loglikelihood), 'kl', np.mean(loss_kldiv))
        if args.output_categorical:
            print(f'mnist accuracy is {100*mnist_correct} %')
        if args.color_clf:
            print(f'color accuracy is {100.*np.mean(color_accuracy)} %')  

        print(time.time())

    return loss_all, loss_recon, loss_recon_loglikelihood, loss_kldiv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    projectdir = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument(
        '--nocontext',
        action="store_true",
        help='overrides the other arguments so that the model is not from context c (ie TEB_0 from y)')
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
        '--use_checkpoint_args',
        type=str2bool,
        default=False,)
    parser.add_argument(
        '--y_stopgradient',
        type=str2bool,
        default=True,
        help='set stopgradient to true to stop gradient flow to Y_model through the latent state c')
    parser.add_argument(
        '--Y_only',
        type=str2bool,
        default=False,
        help = 'train the Y_model only or (train the Y_model first or just start with the full TE model)')
    parser.add_argument(
        '--Y_first',
        type=str2bool,
        default=False,
        help = 'train the Y_model first or just start with the full TE model')
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
    parser.add_argument(
        '--Y_epochs',
        type=int,
        default=20,
        help = 'number of epochs of train_Y')
    parser.add_argument(
        '--TE_epochs',
        type=int,
        default=30,
        help = 'number of epochs of train_TE')
    
    ap_args = parser.parse_args()

    args = getStructuredArgs(f'./{ap_args.config_file}', ap_args)
    
    if args.nocontext:
        print('--nocontext given, overriding args.Y_first, args.Y_continuetrain, args.y_stopgradient so that the model is trained from y and not c')
        args.Y_first = False
        args.Y_continuetrain = False
        args.y_stopgradient = False
    if args.output_categorical:
        args.criterion = TargetLoss(output_type = 'categorical',domain_shape=args.TE_module_args_dict['X_module_args_dict']['input_dim'],presumed_variance=args.presumed_output_variance)
    else:
        args.criterion = TargetLoss(output_type = args.loss_type,domain_shape=args.image_shape,presumed_variance=args.presumed_output_variance)
    if args.loss_type == 'binary':
        args.plot_sigmoid = True
    else:
        args.plot_sigmoid = False
    if args.annealbeta is not False and args.annealbeta != 'none':
        if hasattr(args.beta_TE, "__len__"):
            if len(args.beta_TE) == 2:
                args.startbeta_TE,args.beta_TE = args.beta_TE
            else:
                raise
        else:
            if 'back' in args.annealbeta:
                args.startbeta_TE = args.beta_TE/100.
            else:
                args.startbeta_TE = 100*args.beta_TE 
        
        if isinstance(args.annealbeta, dict):
            print('anneal beta based on dict')
            assert 0 in args.annealbeta.keys(), '0 must be in the keys if using an annealbeta mapping'
            if args.TE_checkpoint>0:
                epochs = np.array(list(args.annealbeta.keys()))
                betaep = epochs[epochs <= args.TE_checkpoint+1].max()
                args.startbeta_TE = args.annealbeta[betaep]
                args.beta_TE = args.annealbeta[betaep]
            else:
                args.startbeta_TE = args.annealbeta[0]
                args.beta_TE = args.annealbeta[0]
        elif 'exponential' in args.annealbeta:
            # or exponentially from 100*beta in 4/5ths TE_epochs steps and then beta for the last 1/5th
            # effective beta then becomes (stepize^epoch)*100*beta for the first 4/5 th's and beta for the last 1/5th
            args.stepsize = (args.startbeta_TE/args.beta_TE)**(5/(4*args.TE_epochs))
        elif 'linear' in args.annealbeta:
            # linear decay from 100*beta in 4/5ths TE_epochs steps and then beta for the last 1/5th
            # linear slope
            args.stepsize = (args.beta_TE-args.startbeta_TE)*(5/(4*args.TE_epochs))
        else:
            #exponential again
            args.stepsize = (args.startbeta_TE/args.beta_TE)**(5/(4*args.TE_epochs))
        
        


    #NOTE: important to note for the paper, that our loss is not directly the loglikelihood, but an average over pixels, but the kl is the true KL, so they need to be brought to the true scale to be traded off between.
    # so true beta for the paper interpretations is this beta/num_pixels.

    #to standardize the above; beta is a tuneable parameter, the rest are computed:
    
    if not args.output_categorical:
        args.normalizing_factor_loglikelihood = np.prod(args.image_shape)
    else:
        args.normalizing_factor_loglikelihood = 1
    args.kappa_Y = args.normalizing_factor_loglikelihood*args.beta_Y
    args.kappa = args.normalizing_factor_loglikelihood*args.beta_TE
    
    if args.color_clf:
        if 'g1_3' in args.testset_argu['directory']:
            args.color_clf_ckpt='ColorCLF.ckpt'
        else:
            raise ValueError('color classifier needed only for needle experiments, dataset folder is expected to have g1_3 in it')
        print(f'Load color classifier from {args.color_clf_ckpt}')
    
    args.exp_name = str(args.exp_name)+'_'+str(args.seed)
    # print arguments
    for arg_name in vars(args):
        print(arg_name, ': ', getattr(args, arg_name))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args_path = f'{args.log_dir}/{args.exp_name}/args.ckpt'

    if args.Y_checkpoint == -1 or args.TE_checkpoint == -1 or not args.use_checkpoint_args:
        if not os.path.exists(f'{args.log_dir}/{args.exp_name}'):
            os.mkdir(f'{args.log_dir}/{args.exp_name}')
        if args.save:
            torch.save(args,args_path)
    else:
        Y_checkpoint = args.Y_checkpoint
        Y_only = args.Y_only
        Y_first = args.Y_first
        Y_epochs = args.Y_epochs
        TE_checkpoint = args.TE_checkpoint
        TE_epochs = args.TE_epochs
        checkpoint_frequency = args.checkpoint_frequency
        Y_module_type = args.Y_module_type
        latent_type = args.latent_type
        sample_c = args.sample_c
        Y_continuetrain = args.Y_continuetrain

        args = torch.load(args_path)
        args.Y_continuetrain = Y_continuetrain

        if Y_module_type != args.Y_module_type:
            raise 'set your Y_module type to be the same as the one you are loading'

        Y_checkpoint = args.Y_checkpoint
        Y_only = args.Y_only
        args.Y_first = Y_first
        args.Y_epochs = Y_epochs
        args.TE_checkpoint = TE_checkpoint
        args.TE_epochs = TE_epochs
        args.checkpoint_frequency = checkpoint_frequency
        args.latent_type = latent_type
        args.sample_c = sample_c
        torch.save(args,args_path)

    print(args)
    if args.Y_only:
        assert not args.Y_first
        Y_module,best_val_epoch_y,best_val_y,train_log_y,val_log_y = main_Y(args)
        print("Y_module Training is done",f'best validation loss is {best_val_y} on epoch {best_val_epoch_y}')
        if args.save:
            torch.save((best_val_epoch_y,train_log_y,val_log_y),f'{args.log_dir}/{args.exp_name}/Y_final_stats.pkl')     
    elif args.Y_first:
        Y_module,best_val_epoch_y,best_val_y,train_log_y,val_log_y = main_Y(args)
        print("Y_module Training is done, moving to training joint TE model",f'best validation loss is {best_val_y} on epoch {best_val_epoch_y}')
        args.Y_checkpoint = args.Y_epochs - 1
        TEmodel, best_TE_ckpt, best_val,train_log,val_log = main_TE(args)
        print(f'Training done, best validation loss is {best_val} on epoch {best_TE_ckpt}')
        if args.save:
            torch.save((best_val,train_log,val_log),f'{args.log_dir}/{args.exp_name}/TE_final_stats.pkl')
    else:
        TEmodel, best_TE_ckpt,best_val,train_log,val_log = main_TE(args)
        print(f'Training done, best validation loss is {best_val} on epoch {best_TE_ckpt}')
        if args.save:
            torch.save((best_val,train_log,val_log),f'{args.log_dir}/{args.exp_name}/TE_final_stats.pkl')