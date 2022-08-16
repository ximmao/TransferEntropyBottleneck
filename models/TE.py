from .misc import init_weight, Detach
from .VAEs import VAEModel, VAEModel_BE, CVAEModel
import numpy as np
import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from typing_extensions import Literal
from TE.utils import onehot

class TEModel(nn.Module):
    """
    Y_module: set to none to initialise a Y_module, otherwise you can pass a Y_module
    share_dec: 0 if we dont do any sharing, 1 if we initialize the variational decoder to be the Y one, 2 if we take the Y decoder and do not train it
    sample_c: whether the model uses the mean and variance of c, or just sampled c. Not for latent_type = 'concat'
    NOTE: key word arguments for initialization take priority over the arg dicts
    """

    def __init__(self, latent_dim=32, Y_module_type=VAEModel_BE, X_module_type=CVAEModel,
                Y_module_args_dict = {'nc':3, 'ndf':64, 'latent_dim':32, 'output_dim':(32, 32),'oc':3, 'dec_multiplier':(2, 2), 'dec_pad':(0,0), 'dec_out_act':'none', 'encoder_type': 'lstm_resnet18_2d','is_2d':True,'output_categorical': False,'teb0_nocontext_mlp_conditionals':False},
                X_module_args_dict = {'input_dim':8, 'ndf':64, 'latent_dim':32, 'output_dim':(32, 32),'oc':3, 'dec_multiplier':(2, 2), 'dec_pad':(0,0), 'dec_out_act':'none', 'latent_type':'concat','sample_c': True, 'encoder_type':'lstm_resnet18_2d','is_2d':True,'mlp_conditionals':False, 'output_categorical': False,'teb0_nocontext_mlp_conditionals':False},
                Y_module: Optional[Type[Union[ VAEModel, VAEModel_BE]]] = None, 
                share_dec: int = 0, x_init_variance=.0000001, x_init_mean_zero = True, output_categorical = False, teb0_nocontext_mlp_conditionals=False, **kwargs):
        super().__init__()
        # overwrite the dict of items for Y module and X module.
        argdict = dict(locals(),**kwargs)
        argdict = {k:v for k,v in argdict.items() if k!='kwargs' and '__' not in k and k!='self' and k!='Y_module_args_dict' and k!='X_module_args_dict'}
        for key,val in argdict.items():
            if key in Y_module_args_dict:
                Y_module_args_dict[key]=val
            if key in X_module_args_dict:
                X_module_args_dict[key]=val
            setattr(self,key,val)
        
        self.teb0_nocontext_mlp_conditionals = teb0_nocontext_mlp_conditionals
        if teb0_nocontext_mlp_conditionals:
            share_dec = 0
        self.output_categorical = output_categorical
        self.share_dec = share_dec
        self.latent_type = X_module_args_dict['latent_type']
        sample_c = X_module_args_dict['sample_c']
        if self.latent_type == 'concat' and not sample_c:
            print('sampling c, since not sampling is redundant for latent_type concat')
            sample_c = True
        self.sample_c = sample_c
        X_module_args_dict['sample_c'] = sample_c
        self.X_module = X_module_type(**X_module_args_dict)
        self.X_module.apply(init_weight)
        if Y_module is not None:
            self.Y_module = Y_module
        else:
            self.Y_module = Y_module_type(**Y_module_args_dict)
            self.Y_module.apply(init_weight)
        self.Y_module_type = type(self.Y_module)
        
        self.x_init_mean_zero = x_init_mean_zero
        self.x_init_variance = x_init_variance
        if x_init_variance != 1:
            # We set the bias of the last layer in the encode to be nonzero and approximately such that the variance in each dimension is x_init_variance at init.
            # Also if x_init_mean_zero, we set the mean to be zero, so no information is transmitted by x at the start
            with torch.no_grad():
                length = X_module_args_dict['latent_dim']
                size_mean = length
                logvar = np.log(x_init_variance)

                if self.X_module.mlp_conditionals:
                    self.X_module.encoder_mlp.l2.bias[size_mean:] = logvar
                    if x_init_mean_zero:
                        self.X_module.encoder_mlp.l2.bias[:size_mean] = 0.
                        self.X_module.encoder_mlp.l2.weight[:,:] = 0.
                else:
                    if self.X_module.encoder_type == 'lstm_resnet18_2d':
                        self.X_module.encoder.proj.bias[size_mean:] = logvar
                        if x_init_mean_zero:
                            self.X_module.encoder.proj.bias[:size_mean] = 0.
                            self.X_module.encoder.proj.weight[:,:] = 0.
                    elif self.X_module.encoder_type == 'resnet18_2d':
                        self.X_module.encoder.fc.bias[size_mean:] = logvar
                        if x_init_mean_zero:
                            self.X_module.encoder.fc.bias[:size_mean] = 0.
                            self.X_module.encoder.fc.weight[:,:] = 0.
                    elif self.X_module.encoder_type == 'lstm_resnet34_2d':
                        self.X_module.encoder.proj.bias[size_mean:] = logvar
                        if x_init_mean_zero:
                            self.X_module.encoder.proj.bias[:size_mean] = 0.
                            self.X_module.encoder.proj.weight[:,:] = 0.
                    elif self.X_module.encoder_type == 'resnet34_2d':
                        self.X_module.encoder.fc.bias[size_mean:] = logvar
                        if x_init_mean_zero:
                            self.X_module.encoder.fc.bias[:size_mean] = 0.
                            self.X_module.encoder.fc.weight[:,:] = 0.
                    elif self.X_module.encoder_type == 'lstm_embed':
                        self.X_module.encoder.proj.bias[size_mean:] = logvar 
                        if x_init_mean_zero:
                            self.X_module.encoder.proj.bias[:size_mean] = 0.
                            self.X_module.encoder.proj.weight[:,:] = 0.
                    elif self.X_module.encoder_type == 'embed':
                        self.X_module.encoder[-1].l2.bias[size_mean:] = logvar
                        if x_init_mean_zero:
                            self.X_module.encoder.l2.bias[:size_mean] = 0.
                            self.X_module.encoder.l2.weight[:,:] = 0.
                    else:
                        self.X_module.encoder.l2.bias[size_mean:] = logvar
                        if x_init_mean_zero:
                            self.X_module.encoder.l2.bias[:size_mean] = 0.
                            self.X_module.encoder.l2.weight[:,:] = 0.
            
        with torch.no_grad():

            # the decoder in the variational module
            if self.share_dec == 0:
                pass
            elif self.share_dec == 1:   
                if self.latent_type == 'concat':
                    raise
                elif self.latent_type == 'add':
                    self.X_module.decoder.load_state_dict(self.Y_module.decoder.state_dict())
            elif self.share_dec == 2:
                if self.latent_type == 'concat':
                    raise
                elif self.latent_type == 'add':

                    self.X_module.decoder.load_state_dict(self.Y_module.decoder.state_dict())
                    self.X_module.decoder.requires_grad = False
                    
        if torch.cuda.is_available():
            self.to(device='cuda')
    
    def forward(self, x, y, y_next, y_next_label = None, stopgradient = True, deterministic=False):
        if self.Y_module_type == VAEModel:
            z_y, yy_reconstructed, y_kl, _y_sample_params = self.Y_module(y, deterministic=deterministic)
        else:
            if self.output_categorical:
                assert y_next_label is not None
                z_y, yy_reconstructed, y_kl, _y_sample_params = self.Y_module(y,y_next_label, deterministic=deterministic)
            else:
                z_y, yy_reconstructed, y_kl, _y_sample_params = self.Y_module(y,y_next, deterministic=deterministic)

        if self.sample_c:
            c = torch.flatten(z_y.detach(),-3)
        else:
            c_mu,c_logvar = _y_sample_params

            if stopgradient:
                c_mu = c_mu.detach()
                c_logvar = c_logvar.detach()
                c = (c_mu,c_logvar)
            else:
                c = (c_mu,c_logvar)
                
            _y_sample_params = c

        if self.X_module.output_categorical:
            assert y_next_label is not None
            zc, y_reconstructed, kl_div, z_params = self.X_module(x, y_next_label, c, deterministic=deterministic)
        else:
            zc, y_reconstructed, kl_div, z_params = self.X_module(x, y_next, c, deterministic=deterministic)

        if deterministic:
            I_z_x_given_c = torch.tensor(0.0)
        elif self.teb0_nocontext_mlp_conditionals:
            actual_c_logvar,_ = _y_sample_params[1].chunk(2, dim=-1)
            I_z_x_given_c = self.X_module.analytical_kl(*z_params, c_mu, actual_c_logvar, reduction= 'mean')
        else:
            I_z_x_given_c = self.X_module.analytical_kl(*z_params, *_y_sample_params, reduction= 'mean')     

        return y_reconstructed, kl_div, z_y, yy_reconstructed, y_kl, I_z_x_given_c #added I_z_x_given_c here

    #decorator disabled because of bug for pytorch 1.6 https://discuss.pytorch.org/t/combining-no-grad-decorator-and-with-torch-no-grad-operator-causes-gradients-to-be-enabled/39203/2
    # @torch.no_grad()
    def info_metrics(self, reconstruction_loglikelihood, reconstruction_loglikelihood_y, kl_div):
        #depricated in favour of the computation in forward, which is for I_z_x_given_c

        # none are upper bounds, and are all approximations. 
        # I_z_x_given_c is an upper bound if I_z_yout_given_c is an upper bound of the actual value
        I_z_yout_given_c = reconstruction_loglikelihood - reconstruction_loglikelihood_y # <log(d_TE(y'|z,c))> - <log(d_ymodel(y'|c))>
        
        I_z_x_given_c = I_z_yout_given_c + kl_div # kl_div=Rate=<log(e(z|x,c))>-<log(b(z|y',c))>

        return I_z_yout_given_c, I_z_x_given_c

    def set_train(self, x_to_train=True, y_to_train=False):
        if x_to_train:
            self.X_module.train()
        else:
            self.X_module.eval()
        if y_to_train:
            self.Y_module.train()
        else:
            self.Y_module.eval()