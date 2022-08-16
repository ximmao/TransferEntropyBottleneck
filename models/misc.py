import torch
import torch.nn as nn
import torchvision
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from typing_extensions import Literal

class ColorClassifier(nn.Module):
    def __init__(self, nc, input_res, oc):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flaten_dim = 4*4*64
        self.fc = nn.Linear(self.flaten_dim, oc)
    
    def forward(self, x):
        bs, _, _, _ = x.size()
        x = self.cnn(x)
        x = x.view(bs, self.flaten_dim)
        out = self.fc(x)
        return out

class Reshape(nn.Module):
    def __init__(self, *args):
        # do not include batch dimension
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # -1 is for batch dimension
        return x.view(-1,*self.shape)

class Detach(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x.detach()

def init_weight(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

def build_grid(resolution): #returns shape (1,resolution[0],resolution[1],4)
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)

class SpatialBroadcastPositionalEncoding(nn.Module):
    def __init__(self,latent_dim,resolution,concat=False):
        """
        if concat is false then the positional embedding is added, otherwise it is concatenated
        """
        super().__init__()
        self.resolution = resolution
        self.concat = concat
        grid = torch.from_numpy(build_grid(resolution)).to(dtype=torch.float) # shape 1,h,w,4
        self.register_buffer("grid", grid)
        self.embed = nn.Linear(4,latent_dim)

    def forward(self,x):
        # shapf of x is .., latent_dim, 1, 1
        # x may be shape batch,.. or batch,seq_len,...
        latent_dim = x.shape[-3]
        target_shape = (*x.shape[:-2],*self.resolution)

        # spatial broadcast
        x=x.expand(*target_shape)

        # postitional embed
        pos = self.embed(self.grid).permute(0,3,1,2) # shape 1,latent_dim,h,w

        if self.concat:
            out = torch.cat(x,pos.repeat(x.shape[0],1,1,1),dim=1)
        else:
            out = x + pos

        return out.view(*target_shape)

class MLP(nn.Module):

    def __init__(self,input_dim,latent_dim,output_dim,p_dropout=0,hidden_activation=nn.ReLU()):
        super().__init__()
        self.l1 = nn.Linear(input_dim,latent_dim)
        self.drop = nn.Dropout(p=p_dropout)
        self.l2 = nn.Linear(latent_dim,output_dim)
        if hidden_activation is None:
            self.hidden_activation = nn.Identity()
        else:
            self.hidden_activation=hidden_activation
    
    def forward(self,x):
        x = self.l1(x)
        x = self.drop(x)
        x = self.hidden_activation(x)
        x = self.l2(x)
        return x

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_type, emb_inputchannels, is_2d=True, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.emb_type = emb_type

        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0, bidirectional=self.bidirectional)

        #need our projection to have a bias
        self.proj = nn.Linear(hidden_size,output_size)

        self.is_2d = is_2d
        if self.is_2d:
            self.enc,_ = select_resnet(emb_type, input_size, nc = emb_inputchannels)
        elif emb_type == 'embed':
            self.enc = nn.Sequential(nn.Embedding(emb_inputchannels,hidden_size),MLP(hidden_size,hidden_size,input_size))
    
    def forward(self, x):
        if self.is_2d:
            batch_size, seq_len, ch, res1, res2 = x.size()
            assert res1 == res2
            res = res1
            x = x.reshape(batch_size*seq_len, ch, res, res)
        else:
            batch_size, seq_len, dim = x.size()
            x = x.view(batch_size*seq_len, dim)
            if self.emb_type == 'embed':
                x = x.squeeze(-1)
            
        x = self.enc(x)
        _, out_dim = x.size()
        assert out_dim == self.input_size

        x = x.view(batch_size, seq_len, self.input_size)

        h0, c0 = self.initHidden(x.size(0))
        h0, c0 = h0.to(x.device), c0.to(x.device)
        out, (hk, ck) = self.lstm(x, (h0, c0))
        if self.bidirectional:
            out = out.view(batch_size, seq_len, 2, self.output_size) 
            last = out[:, -1, :, :]
            last = torch.mean(last, 1)
        else:
            last = out[:, -1, :]
        last = self.proj(last)

        return last
        
    def initHidden(self, batch_size):
        s = 2 if self.bidirectional else 1 
        return (torch.zeros(self.num_layers*s, batch_size, self.hidden_size), torch.zeros(self.num_layers*s, batch_size, self.hidden_size))

class VectorField(nn.Module):
    def __init__(self, latent_dim, hidden_dim=50):
        super().__init__()
        self.model = nn.Sequential(
                      nn.Linear(atent_dim, hidden_dim),
                      nn.ELU(),
                      nn.Linear(hidden_dim, hidden_dim),
                      nn.ELU(),
                      nn.Linear(hidden_dim, atent_dim))

    def forward(self, t, x):
        return self.model(x)

class VFDecoder(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dim=50):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim))

    def forward(self, x):
        return self.model(x)        

def select_resnet(network, output_dim=128, nc =3, track_running_stats=True):
    # select 2d resnet from torchvision, for TE_var models. note this resnet has fc layer. 
    # Note input channels is 3, to change it, reset model.conv1 to what you want eg: model.conv1 = nn.Conv2d(input_channels, model.inplanes, kernel_size=7, stride=2, padding=3,bias=True)
    param = {'feature_size': 1024}
    if network == 'resnet18_2d':
        model = torchvision.models.resnet18(num_classes=output_dim,zero_init_residual=True)
        if nc!=3:
            model.conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3,
                                bias=True) #hack the first layer to have correct number of input channels
        param['feature_size'] = 256 # TODO, check final feature size
    elif network == 'resnet34_2d':
        if nc !=3:
            raise NotImplementedError()
        else:
            model = torchvision.models.resnet34(num_classes=output_dim,zero_init_residual=True)
    return model, param