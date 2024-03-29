# from cv2 import multiply
from .VAE_abs import VAEModel_abs
from .misc import *
import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from typing_extensions import Literal

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

class VAEModel(VAEModel_abs):
    def __init__(self, nc=1, ndf=64, latent_dim=32, output_dim=(32, 32), bilstm=False,
                 oc=1, dec_multiplier=(2, 2), dec_pad=(0,0), dec_out_act='tanh', seq_len_out=1,
                 is_2d=True, encoder_type='resnet18_2d', use_neural_ode=False, output_seq_scalar=False,
                 include_first_in_target = False, dec_fs = 1., y_input_dim=1, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'lstm_resnet18_2d':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'resnet18_2d', emb_inputchannels = nc, is_2d = is_2d, bidirectional=bilstm)
        elif encoder_type == 'lstm_resnet34_2d':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'resnet34_2d', emb_inputchannels = nc, is_2d = is_2d, bidirectional=bilstm)
        elif encoder_type == 'lstm_scalar':
            lstm_input_size=y_input_dim
            lstm_hidden_size=latent_dim*2
            lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'identity', emb_inputchannels = 1, is_2d = False, bidirectional=bilstm)
        else:
            # encoder resnet instead
            self.encoder,_ = select_resnet(encoder_type, latent_dim*2,nc=nc) 
        self.output_seq_scalar = output_seq_scalar
        if self.output_seq_scalar:
            # assert use_neural_ode
            if not use_neural_ode:
                if encoder_type == 'lstm_scalar':
                    print('baseline LSTM: use MLP as decoder')
                else:
                    raise NotImplementedError()
        self.use_neural_ode = use_neural_ode
        self.seq_len_out = seq_len_out
        self.include_first_in_target = include_first_in_target
        self.dec_fs = dec_fs
        if use_neural_ode and encoder_type == 'lstm_scalar':
            self.decoder = VFDecoder(oc, latent_dim)
            self.f = VectorField(latent_dim)
        elif output_seq_scalar:
            if self.include_first_in_target:
                mlp_output_dim = seq_len_out + 1
            else:
                mlp_output_dim = seq_len_out
            assert oc == 1
            self.decoder = MLP(latent_dim, latent_dim, mlp_output_dim)        
        else:
            decoder_func = self.get_decoder_func(is_2d=is_2d, oc=oc, latent_dim=latent_dim, output_dim=output_dim, 
                                             ndf=ndf, dec_pad=dec_pad, dec_multiplier=dec_multiplier, 
                                             act_output=dec_out_act)
            self.decoder = decoder_func()
        # def run_neuralODE_decoder():
        #     pred_z = odeint(f, sampled_ode, batch_t) # sample_traj_length(batch_t dim) x bs x latent_dim  
        #     pred_z = pred_z.permute(1, 0, 2) # bs x sample_traj_length x latent_dim  
        #     pred_traj = dec(pred_z) # bs x sample_traj_length x state_dim
        # return run_neuralODE_decoder
    
    def forward(self, y, deterministic=False):
        if self.encoder_type == 'lstm_scalar':
            batch_size, seq_len, _ = y.size()
        else:
            batch_size, seq_len, ch, _, _ = y.size()
        y = self.encoder(y)

        mu, logvar = y.chunk(2, dim=-1)
        
        if not deterministic: 
            z = self.posterior_sample(torch.randn_like(logvar), mu, logvar)
            kl_div = self.analytical_kl(mu, logvar, torch.zeros_like(mu), torch.zeros_like(logvar))
        else:
            z = mu
            kl_div = torch.tensor(0.0)       

        z = z.view(batch_size, -1, 1, 1)
        if self.output_seq_scalar and self.use_neural_ode:
            if self.include_first_in_target:
                t_pred = torch.linspace(0./self.dec_fs, float(self.seq_len_out)/self.dec_fs, int(self.seq_len_out)+1)
            else:
                t_pred = torch.linspace(1./self.dec_fs, float(self.seq_len_out)/self.dec_fs, int(self.seq_len_out))
            z_out = odeint(self.f, z.view(batch_size,-1), t_pred).permute(1, 0, 2) # bs x len x 1
            y_out = self.decoder(z_out) # y_out size is bs x len x 1
        else:
            if self.output_seq_scalar:
                y_out = self.decoder(z.view(batch_size,-1))
                y_out = y_out.unsqueeze(2)
            else:
                y_out = self.decoder(z)

        return z, y_out, kl_div, (mu,logvar)

class VAEModel_BE(VAEModel_abs):
    def __init__(self, nc=1, ndf=64, latent_dim=32, output_dim=(32, 32), bilstm=False,
                 oc=1, dec_multiplier=(2, 2), dec_pad=(0,0), dec_out_act='none', seq_len_out=1,
                 is_2d=True, encoder_type='lstm_resnet18_2d',output_categorical = False,teb0_nocontext_mlp_conditionals=False,
                 use_neural_ode=False, output_seq_scalar=False, include_first_in_target = False, dec_fs = 1., y_input_dim = 1, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim # added since teb0_nocontext_mlp_conditionals accesses it
        self.output_categorical = output_categorical
        if output_categorical:
            # assume the input and putput are the same space for now
            encoder_type = 'embed'

        self.teb0_nocontext_mlp_conditionals = teb0_nocontext_mlp_conditionals
        self.encoder_type = encoder_type
        self.output_seq_scalar = output_seq_scalar
        if self.output_seq_scalar:
            assert use_neural_ode
        self.use_neural_ode = use_neural_ode
        self.seq_len_out = seq_len_out
        self.include_first_in_target = include_first_in_target
        self.dec_fs = dec_fs
        if encoder_type == 'lstm_resnet18_2d':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            if teb0_nocontext_mlp_conditionals:
                lstm_output_size=latent_dim*3
            else:
                lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'resnet18_2d', emb_inputchannels = nc, is_2d = is_2d, bidirectional=bilstm)
            backwards_network = encoder_type[5:]
        elif encoder_type == 'lstm_resnet34_2d':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            if teb0_nocontext_mlp_conditionals:
                lstm_output_size=latent_dim*3
            else:
                lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'resnet34_2d', emb_inputchannels = nc, is_2d = is_2d, bidirectional=bilstm)
            backwards_network = encoder_type[5:]
        elif encoder_type == 'lstm_embed':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            if teb0_nocontext_mlp_conditionals:
                lstm_output_size=latent_dim*3
            else:
                lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'embed', emb_inputchannels = nc, is_2d = False, bidirectional=bilstm)
        elif encoder_type == 'lstm_scalar':
            lstm_input_size=y_input_dim
            lstm_hidden_size=latent_dim*2
            if teb0_nocontext_mlp_conditionals:
                lstm_output_size=latent_dim*3
            else:
                lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'identity', emb_inputchannels = 1, is_2d = False, bidirectional=bilstm)
        elif encoder_type == 'embed':
            if teb0_nocontext_mlp_conditionals:
                self.encoder = nn.Sequential(nn.Embedding(nc,latent_dim),MLP(latent_dim,latent_dim,latent_dim*3))
            else:
                self.encoder = nn.Sequential(nn.Embedding(nc,latent_dim),MLP(latent_dim,latent_dim,latent_dim*2))
        
        else:
            # encoder resnet instead
            if teb0_nocontext_mlp_conditionals:
                self.encoder,_ = select_resnet(encoder_type, latent_dim*3,nc=nc)
            else:
                self.encoder,_ = select_resnet(encoder_type, latent_dim*2,nc=nc)
            backwards_network = encoder_type

        if self.use_neural_ode and encoder_type == 'lstm_scalar':
            self.decoder = VFDecoder(oc, latent_dim)
            self.f = VectorField(latent_dim)
            self.backwards_enc = LSTM_Encoder(oc, lstm_hidden_size, lstm_output_size, 'identity', emb_inputchannels = 1, is_2d = False, bidirectional=bilstm)
        elif output_categorical:
            self.decoder = MLP(latent_dim,latent_dim,nc)
            self.backwards_enc = nn.Sequential(nn.Embedding(oc,latent_dim),MLP(latent_dim,latent_dim,latent_dim*2))
        else:
            decoder_func = self.get_decoder_func(is_2d=is_2d, oc=oc, latent_dim=latent_dim, output_dim=output_dim, 
                                                ndf=ndf, dec_pad=dec_pad, dec_multiplier=dec_multiplier, 
                                                act_output=dec_out_act)
            self.decoder = decoder_func()
            self.backwards_enc,_ = select_resnet(backwards_network,latent_dim*2,nc=oc)
            
    def forward(self, y, y_next, deterministic=False):
        
        y_next = y_next.clone()
        if self.encoder_type == 'embed':
            assert len(y.size()) == 1
            batch_size = y.size()[0]
        elif self.encoder_type == 'lstm_scalar':
            batch_size, seq_len, _ = y.size()
        else:
            batch_size, seq_len, ch, _, _ = y.size()

        y = self.encoder(y)
        if self.teb0_nocontext_mlp_conditionals:
            # first half of logvar is logvar, second half is an additional ouput for conditioning
            mu, logvar = torch.split(y, split_size_or_sections=[self.latent_dim, 2*self.latent_dim], dim=-1)
        else:
            mu, logvar = y.chunk(2, dim=-1)

        if not deterministic and not self.teb0_nocontext_mlp_conditionals:
            # get backwards encoding for kl div:
            if self.use_neural_ode: # means that it is sequential output
                assert len(y_next.size()) == 3
            else:
                seq_len_next = y_next.size()[1]
                assert seq_len_next == 1
                y_next = y_next.squeeze(1)
            back = self.backwards_enc(y_next)

            mu_back, logvar_back = back.chunk(2, dim=-1)

            kl_div = self.analytical_kl(mu, logvar, mu_back, logvar_back)

            z = self.posterior_sample(torch.randn_like(logvar), mu, logvar)
        else:
            kl_div = torch.tensor(0.0)
            z = mu

        z = z.view(batch_size, -1, 1, 1)
        
        if self.output_seq_scalar and self.use_neural_ode:
            if self.include_first_in_target:
                t_pred = torch.linspace(0./self.dec_fs, float(self.seq_len_out)/self.dec_fs, int(self.seq_len_out)+1)
            else:
                t_pred = torch.linspace(1./self.dec_fs, float(self.seq_len_out)/self.dec_fs, int(self.seq_len_out))
            z_out = odeint(self.f, z.view(batch_size,-1), t_pred).permute(1, 0, 2) # bs x len x 1
            y_out = self.decoder(z_out) # y_out size is bs x len x 1
        elif self.output_categorical:
            y_out = self.decoder(z.view(batch_size,-1))
        else:
            y_out = self.decoder(z)
        return z, y_out, kl_div, (mu,logvar)

class CVAEModel(VAEModel_abs):
    def __init__(self, input_dim=1, ndf=64, latent_dim=32, output_dim=(32,32), bilstm=False,
                 oc=1, dec_multiplier=(2, 2), dec_pad=(0,0), dec_out_act='none', seq_len_out=1,
                 is_2d=True, latent_type: Literal['add','concat'] = 'concat',
                 sample_c = True, encoder_type = 'lstm_resnet18_2d', 
                 mlp_conditionals=False, output_categorical = False, teb0_nocontext_mlp_conditionals=False, 
                 use_neural_ode=False, output_seq_scalar = False, include_first_in_target = False, dec_fs = 1., **kwargs):
        super().__init__()

        self.latent_type = latent_type
        self.input_dim = input_dim
        self.mlp_conditionals = mlp_conditionals
        self.bilstm =bilstm
        self.output_categorical = output_categorical
        self.teb0_nocontext_mlp_conditionals = teb0_nocontext_mlp_conditionals
        self.output_seq_scalar = output_seq_scalar
        if self.output_seq_scalar:
            assert use_neural_ode
        self.use_neural_ode = use_neural_ode
        self.seq_len_out = seq_len_out
        self.include_first_in_target = include_first_in_target
        self.dec_fs = dec_fs
        if teb0_nocontext_mlp_conditionals:
            # first half of logvar is logvar, second half is an additional ouput for conditioning
            assert mlp_conditionals
            assert not sample_c
        if output_categorical:
            encoder_type = 'resnet18_2d'
        if self.latent_type == 'concat':
            dec_latent = latent_dim
            if not sample_c:
                print('sampling c, since not sampling is redundant for latent_type concat')
                sample_c = True
        elif self.latent_type == 'add':
            dec_latent = latent_dim
        else:
            raise f'latent_type of {self} must be concat or add'

        self.sample_c = sample_c
        self.encoder_type = encoder_type

        if output_categorical:
            self.encoder,_ = select_resnet(encoder_type, latent_dim*2,nc=oc)
        elif encoder_type == 'lstm_resnet18_2d':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'resnet18_2d', emb_inputchannels = input_dim, is_2d = is_2d, bidirectional=bilstm)
        elif encoder_type == 'resnet18_2d' or encoder_type == 'resnet34_2d':
            # encoder resnet instead
            self.encoder,_ = select_resnet(encoder_type, latent_dim*2,nc=input_dim)
        elif encoder_type == 'lstm_resnet34_2d':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'resnet34_2d', emb_inputchannels = input_dim, is_2d = is_2d, bidirectional=bilstm)
        elif encoder_type == 'lstm_embed':
            lstm_input_size=latent_dim*2
            lstm_hidden_size=latent_dim*2
            lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'embed', emb_inputchannels = input_dim, is_2d = False, bidirectional=bilstm)
        elif encoder_type == 'embed':
            self.encoder = nn.Sequential(nn.Embedding(input_dim,latent_dim),MLP(latent_dim,latent_dim,latent_dim*2))
        elif encoder_type == 'lstm_scalar':
            lstm_input_size=input_dim
            lstm_hidden_size=latent_dim*2
            lstm_output_size=latent_dim*2
            self.encoder = LSTM_Encoder(lstm_input_size, lstm_hidden_size, lstm_output_size, 'identity', emb_inputchannels = 1, is_2d = False, bidirectional=bilstm)
        else:
            self.encoder = MLP(input_dim,latent_dim,latent_dim*2,p_dropout=0, hidden_activation=nn.ReLU())


        if (self.output_seq_scalar and self.use_neural_ode) and encoder_type == 'lstm_scalar':
            self.decoder = VFDecoder(oc, latent_dim)
            self.f = VectorField(latent_dim)
        elif output_categorical:
            self.decoder = MLP(latent_dim,latent_dim,input_dim)
        else:
            decoder_func = self.get_decoder_func(is_2d=is_2d, oc=oc, latent_dim=dec_latent, output_dim=output_dim, 
                                                ndf=ndf, dec_pad=dec_pad, dec_multiplier=dec_multiplier, 
                                                act_output=dec_out_act)
            self.decoder = decoder_func()

        if mlp_conditionals:
            if sample_c:
                self.encoder_mlp = MLP(latent_dim*3,latent_dim,latent_dim*2,p_dropout=0, hidden_activation=nn.ReLU())
            elif teb0_nocontext_mlp_conditionals:
                # first half of logvar is logvar, second half is an additional ouput for conditioning
                self.encoder_mlp = MLP(latent_dim*3,latent_dim,latent_dim*2,p_dropout=0, hidden_activation=nn.ReLU())
            else:
                self.encoder_mlp = MLP(latent_dim*4,latent_dim,latent_dim*2,p_dropout=0, hidden_activation=nn.ReLU())
        
    def forward(self, x, y_next, c = None, y = None, sequential = False, deterministic=False):
        """
        c is the context latent, if computed elsewhere
        """

        if c is None:
            raise 

        y_next = y_next.clone()

        if self.encoder_type == 'lstm_resnet18_2d' or self.encoder_type == 'lstm_resnet34_2d':
            batch_size, seq_len, ch, _, _ = x.size()
        elif self.encoder_type == 'resnet18_2d' or self.encoder_type == 'resnet34_2d':
            batch_size, ch, _, _ = x.size()
        elif self.encoder_type == 'lstm_embed':
            batch_size, seq_len = x.size()
            x = x.unsqueeze(-1).contiguous()
        elif self.encoder_type == 'embed':
            assert len(x.size()) == 1
            batch_size = x.size()[0]
        elif self.encoder_type == 'lstm_scalar':
            batch_size, seq_len, _ = x.size()
        else:
            batch_size, x_dim = x.size()

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        if self.mlp_conditionals:
            if self.sample_c:
                x = self.encoder_mlp(torch.cat([x,c],dim=-1))
            elif self.teb0_nocontext_mlp_conditionals:
                c_mu,c_extra = c[0],c[1]
                c_logvar,ycondition = c_extra.chunk(2, dim=-1)
                x = self.encoder_mlp(torch.cat([x,ycondition],dim=-1))
            else:
                c_mu,c_logvar = c[0],c[1]
                x = self.encoder_mlp(torch.cat([x,c_mu,c_logvar],dim=-1))
            
        mu, logvar = x.chunk(2, dim=-1)
        
        if not self.sample_c:
            if not self.teb0_nocontext_mlp_conditionals:
                c_mu,c_logvar = c[0],c[1]
            if self.latent_type == 'add':
                mu = mu + c_mu
            else:
                raise 'its a bug that you got here'
        
        z = self.posterior_sample(torch.randn_like(logvar), mu, logvar)
        z = z.view(batch_size, -1, 1, 1)
        if self.sample_c:
            c = c.view(batch_size, -1, 1, 1)

            if self.latent_type == 'concat':
                if deterministic:
                    zc = torch.cat([mu,c],dim=1)
                else:
                    zc = torch.cat([z,c],dim=1)
            elif self.latent_type == 'add':
                if deterministic:
                    zc = (mu.view(batch_size, -1, 1, 1) + c)
                else:
                    zc = z + c
        else:
            if deterministic:
                zc = mu.view(batch_size, -1, 1, 1)
            else:
                zc = z
        
        if self.output_seq_scalar and self.use_neural_ode:
            if self.include_first_in_target:
                t_pred = torch.linspace(0./self.dec_fs, float(self.seq_len_out)/self.dec_fs, int(self.seq_len_out)+1)
            else:
                t_pred = torch.linspace(1./self.dec_fs, float(self.seq_len_out)/self.dec_fs, int(self.seq_len_out))
            zc_out = odeint(self.f, zc.view(batch_size,-1), t_pred).permute(1, 0, 2) # bs x len x latent
            out = self.decoder(zc_out) # y_out size is bs x len x oc
        elif self.output_categorical:
            out = self.decoder(zc.view(batch_size,-1))
        else:
            out = self.decoder(zc)
        
        kl_div = torch.tensor(0.0)

        return zc, out, kl_div, (mu,logvar)