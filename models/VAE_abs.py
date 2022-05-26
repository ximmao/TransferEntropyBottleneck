import torch
import torch.nn as nn
from functools import partial
from .misc import SpatialBroadcastPositionalEncoding

class VAEModel_abs(nn.Module):

    def posterior_sample(self, z, mu_q, logvar_q):
        return torch.mul(torch.exp(logvar_q / 2.), z) + mu_q

    def analytical_kl(self, mu_q, logvar_q, mu_p, logvar_p, reduction = 'mean'):
        """analytical kl between two fully factorized multi-dimensional gaussian. reduction is for reduction along the batch dimension """
        batch_size = mu_q.size(0)
        assert batch_size == logvar_q.size(0), 'batch size mismatch mu_q and logvar_q'
        assert mu_q.size(1) == logvar_q.size(1)
        assert batch_size == mu_p.size(0), 'batch size mismatch mu_q and mu_p'
        assert batch_size == logvar_p.size(0), 'batch size mismatch mu_q and logvar_p'

        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        
        if reduction == 'mean':
            return torch.mean(torch.sum(0.5*(var_q/var_p + torch.pow(mu_q - mu_p, 2)/var_p - 1. + logvar_p - logvar_q), dim=-1),dim=0)
        elif reduction == 'sum':
            return torch.sum(torch.sum(0.5*(var_q/var_p + torch.pow(mu_q - mu_p, 2)/var_p - 1. + logvar_p - logvar_q), dim=-1),dim=0)
        elif reduction == 'max':
            return torch.max(torch.sum(0.5*(var_q/var_p + torch.pow(mu_q - mu_p, 2)/var_p - 1. + logvar_p - logvar_q), dim=-1),dim=0)
        elif reduction == 'min':
            return torch.min(torch.sum(0.5*(var_q/var_p + torch.pow(mu_q - mu_p, 2)/var_p - 1. + logvar_p - logvar_q), dim=-1),dim=0)
        elif reduction == 'none':
            return torch.sum(0.5*(var_q/var_p + torch.pow(mu_q - mu_p, 2)/var_p - 1. + logvar_p - logvar_q), dim=-1)
        else:
            raise
    
    def define_dec_conv2d_block(self, sf, pb, ci, co, k, s, p, act='relu', batchnorm=True):
        # resize-conv block to get rid of checkerboard effect
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
        # output_dim = floor(((input_dim * sf + 2) + 2p - d(k - 1) - 1)/s + 1)

        model = [nn.ReflectionPad2d(pb),
                nn.Conv2d(ci, co, kernel_size=k, stride=s, padding=p, bias=False)]

        if batchnorm:
            model += [nn.BatchNorm2d(co)]
        if act == 'relu':
            model += [nn.ReLU(True)]
        elif act == 'lrelu':
            model += [nn.LeakyReLU(0.2, True)]
        elif act == 'elu':
            model += [nn.ELU(True)]
        elif act == 'tanh':
            model += [nn.Tanh()]
        elif act == 'sigmoid':
            model += [nn.Sigmoid()]
        elif act == 'none':
            pass
        
        return nn.Sequential(*model)

    def get_n_block(self, multiplier, start, end, pad):
        # return the minimum number of blocks that can obtain the desired output dimension
        d = {}
        for i in range(2):
            d[i] = {'num':None, 'res':None}
            if start[i] >= end[i]:
                d[i]['num'] = int(0)
                d[i]['res'] = int(end[i] - start[i])
            else:
                curr = start[i]
                cnt = 0
                while curr < end[i]:
                    curr = curr * multiplier[i] + 2*pad[i]
                    cnt += 1
                d[i]['num'] = int(cnt)
                d[i]['res'] = int(end[i] - curr)
        assert d[0]['res'] <= 0
        assert d[1]['res'] <= 0
        return d
    
    def get_decoder_func(self, is_2d, oc, latent_dim, output_dim, ndf, dec_pad=(0,0), dec_multiplier=(2, 2), act_output='tanh'):
        if is_2d:
            return partial(self.define_decoder_2d, oc=oc, latent_dim=latent_dim, output_dim=output_dim, 
                           ndf=ndf, dec_pad=(0,0), dec_multiplier=(2, 2), act_output=act_output)
        else:
            return self.define_decoder_1d


    def define_decoder_2d(self, oc, latent_dim, output_dim, ndf=64, dec_pad=(0,0), dec_multiplier=(2, 2), act_output='tanh'):
        # each resize-conv block scale each axis by scaling_factor and reflectionpad by 1 before apply conv operation
        # assume 2d square output
        idx_layer = 1
        dec_params = self.get_n_block(dec_multiplier, (4, 4), output_dim, dec_pad)
        print(dec_params)
        smaller_dim = int(0) if dec_params[0]['num'] < dec_params[1]['num'] else int(1)
        num_dec_layer_min = dec_params[smaller_dim]['num']
        num_dec_layer_max = dec_params[int(1-smaller_dim)]['num']
        assert num_dec_layer_max == num_dec_layer_min
        res = dec_params[0]['res']
        
        decoder = [SpatialBroadcastPositionalEncoding(latent_dim,resolution = output_dim)]  

        decoder += [self.define_dec_conv2d_block((1, 1), (dec_pad[1], dec_pad[1], dec_pad[0], dec_pad[0]),
                                                       latent_dim,
                                                       int(ndf*(2**(num_dec_layer_max-idx_layer-1))),
                                                       7, 1, 3, 'relu')]

        idx_layer += 1
        for k in range(idx_layer, idx_layer + num_dec_layer_min):
            print(dec_multiplier, (dec_pad[1], dec_pad[1], dec_pad[0], dec_pad[0]), int(ndf*(2**(num_dec_layer_max+1-k-1))), int(ndf*(2**(num_dec_layer_max-k-1))))
            decoder += [self.define_dec_conv2d_block(dec_multiplier, (dec_pad[1], dec_pad[1], dec_pad[0], dec_pad[0]),
                                                            int(ndf*(2**(num_dec_layer_max+1-k-1))), 
                                                            int(ndf*(2**(num_dec_layer_max-k-1))), 
                                                            5, 1, 2, 'relu')]
        idx_layer += (num_dec_layer_min - 1)
        curr_ch = int(ndf*(2**(num_dec_layer_max-idx_layer-1)))

        if res < 0:
            for j in range((-res)//2):
                decoder += [nn.Conv2d(curr_ch, int(max(curr_ch//2, oc)), kernel_size=3, stride=1, padding=1, bias=False)]
                curr_ch = int(max(curr_ch//2, oc))
            decoder += [nn.BatchNorm2d(curr_ch),
                       nn.ReLU(True)]

        decoder += [nn.Conv2d(curr_ch, oc, kernel_size=3, stride=1, padding=1, bias=False)]
        if act_output == 'tanh':
            decoder += [nn.Tanh()]
        elif act_output == 'sigmoid':
            decoder += [nn.Sigmoid()]
        else:
            assert act_output == 'none'

        return nn.Sequential(*decoder)

    def define_decoder_1d(self, oc, latent_dim, output_dim, ndf=64, dec_pad=(0,0), dec_multiplier=(2, 2), act_output='tanh'):
        # each resize-conv block scale each axis by scaling_factor and reflectionpad by 1 before apply conv operation
        # assume 2d square output
        idx_layer = 1
        dec_params = self.get_n_block(dec_multiplier, (4, 4), output_dim, dec_pad)
        print(dec_params)
        smaller_dim = int(0) if dec_params[0]['num'] < dec_params[1]['num'] else int(1)
        num_dec_layer_min = dec_params[smaller_dim]['num']
        num_dec_layer_max = dec_params[int(1-smaller_dim)]['num']
        assert num_dec_layer_max == num_dec_layer_min
        res = dec_params[0]['res']
        
        decoder = [SpatialBroadcastPositionalEncoding(latent_dim,resolution = output_dim)] 
        decoder += [self.define_dec_conv2d_block((4, 4), (dec_pad[1], dec_pad[1], dec_pad[0], dec_pad[0]),
                                                       latent_dim,
                                                       int(ndf*(2**(num_dec_layer_max-idx_layer-1))),
                                                       7, 1, 3, 'relu')]

        idx_layer += 1
        for k in range(idx_layer, idx_layer + num_dec_layer_min):
            print(dec_multiplier, (dec_pad[1], dec_pad[1], dec_pad[0], dec_pad[0]), int(ndf*(2**(num_dec_layer_max+1-k-1))), int(ndf*(2**(num_dec_layer_max-k-1))))
            decoder += [self.define_dec_conv2d_block(dec_multiplier, (dec_pad[1], dec_pad[1], dec_pad[0], dec_pad[0]),
                                                            int(ndf*(2**(num_dec_layer_max+1-k-1))), 
                                                            int(ndf*(2**(num_dec_layer_max-k-1))), 
                                                            5, 1, 2, 'relu')]
        idx_layer += (num_dec_layer_min - 1)
        curr_ch = int(ndf*(2**(num_dec_layer_max-idx_layer-1)))

        if res < 0:
            for j in range((-res)//2):
                decoder += [nn.Conv2d(curr_ch, int(max(curr_ch//2, oc)), kernel_size=3, stride=1, padding=1, bias=False)]
                curr_ch = int(max(curr_ch//2, oc))
            decoder += [nn.BatchNorm2d(curr_ch),
                       nn.ReLU(True)]

        decoder += [nn.Conv2d(curr_ch, oc, kernel_size=3, stride=1, padding=1, bias=False)]
        if act_output == 'tanh':
            decoder += [nn.Tanh()]
        else:
            assert act_output == 'none'

        return nn.Sequential(*decoder)