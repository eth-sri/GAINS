import numpy as np
import os

import torch 
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import argparse
import pdb
from torchdiffeq import odeint_adjoint
import torchdiffeq._impl.odeint as odeint
#from anode.adjoint import odesolver_adjoint as odeint_anode

from scipy.stats import ortho_group

class Normalization(nn.Module):
    #Normalization layer 
    def __init__(self,device, mean,std):
        super(Normalization, self).__init__()
        if len(mean) == 1:
            self.mean = torch.FloatTensor([mean]).view((1, 1, 1, 1)).to(device)
            self.sigma = torch.FloatTensor([std]).view((1, 1, 1, 1)).to(device)
        elif len(mean) ==3:
            self.mean = torch.FloatTensor([mean]).view((1, 3, 1, 1)).to(device)
            self.sigma = torch.FloatTensor([std]).view((1, 3, 1, 1)).to(device)
        else:
            print("Norm not implemented for length:",len(mean))
            exit()

    def forward(self, x):
        return (x - self.mean) / self.sigma

def norm(dim,nogr=32):
    return nn.GroupNorm(min(nogr, dim), dim)

def norm_LN(dim):
    return nn.InstanceNorm2d(dim, affine = True)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class BatchNorm(nn.Module):
    def __init__(self,num_features,device,n_max_tp,endtime):
        super(BatchNorm, self).__init__()

        self.n_max_tp = n_max_tp
        self.device = device
        self.endtime = endtime
        self.num_features = num_features
        
        if n_max_tp % 2 == 1:
            equi_dist = endtime / (n_max_tp - 1)
        else:
            equi_dist = endtime / n_max_tp 
        #init BN-layer statistics at equidistant timepoint
        self.timepoints = torch.zeros(n_max_tp).to(device)
        seq = []
        for i in range(n_max_tp):
            self.timepoints[i] = i * equi_dist
            seq.append(nn.BatchNorm2d(num_features).to(device))
        self.BatchNorm2d = nn.Sequential(*seq)

    def forward(self,t, x):
        key= (self.timepoints - t).abs().argmin()
        out= self.BatchNorm2d[key](x)
        return out
#-********************************************
# CNN
class ResBlock(nn.Module):
    def __init__(self, in_planes, planes,act, stride=1):
        super(ResBlock, self).__init__()
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act =="sin":
            self.act = torch.sin
        else:
            print("not implemented activation for baseline")
            exit()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm(in_planes)
        #self.norm1 = nn.BatchNorm2d(in_planes)
        #self.norm1 = norm_LN(in_planes)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.norm2 = norm(planes)
        self.norm2 = norm(in_planes)
        #self.norm2 = norm_LN(planes)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        #out = self.norm2(out)
        out = self.act(out)
        return out + x

class CNN_mini_MNIST(nn.Module):
    def __init__(self,device,norm_mean,norm_std, act = "relu",layers=1,shared = True):
        super(CNN_mini_MNIST, self).__init__()
        self.act = act
        self.device = device
        normalize_MNIST = [Normalization(device,norm_mean,norm_std)]

        downsampling_layers = [
            nn.Conv2d(1, 32, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, 2),
            nn.ReLU(inplace=True)
            ]
        if shared:
            residual_layer = ResBlock(32, 32,self.act)
            feature_layers = [residual_layer for _ in range(layers)]
        else:
            feature_layers = [ResBlock(32, 32,self.act) for _ in range(layers)]
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(32, 10)
            ]
        self.net = nn.Sequential(*normalize_MNIST,*downsampling_layers, *feature_layers, *fc_layers)

    def forward(self, x):
        return self.net(x)

#-**********************************************************************************-#
# ODE models

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)
        
    def forward(self, t, x):
        return self._layer(torch.cat([torch.ones_like(x[:, :1, :, :]) * t, x], 1))

class ODEBlock(nn.Module):
    def __init__(self, odefunc, met= "dopri5" ,adjoint = 0 ,t=[0,1]): #maybe try different time
        #odefunc is an instance of ODEfunc, which contains the "mini-network" that describes the ODE function
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor(t).float()
        self.met = met
        self.method = met
        self.atol = 1e-9 #default value
        self.rtol= 1e-7  #default value
        self.adjoint = adjoint
        self.step_size = 0.1 # default value
        self.running_mean_step = None
        self.trajectory_path = None
        self.last_step = None
        if adjoint == 0: #store forward passes
            self.odesolver = odeint
            self.anode = 0
        if adjoint == 3: #store forward passes
            self.odesolver = odeint
            self.anode = 3
        elif adjoint == 1: #adjoint method
            self.odesolver = odeint_adjoint
            self.anode = 1

        elif adjoint == 2: #ANODE checkpointing same as 0 if only one ODeblock
            self.odesolver = odeint_anode
            self.options = {}
            self.options.update({'Nt':int(2)})
            self.options.update({'method':met})
            self.anode = 2
        if met == "euler":
            self.met = "euler"
            self.step_size = 1.0

        if met == "dopri5_0.1": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.1
            self.rtol = 0.1
        if met == "dopri5_0.01": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.01
            self.rtol = 0.00
        if met == "dopri5_0.005": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.005
            self.rtol = 0

        if met == "dopri5_0.00005": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.00005
            self.rtol = 0

        if met == "dopri5_0.001": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001
            self.rtol = 0
        if met == "dopri5_0.0001": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.0001
            self.rtol = 0

        if met == "dopri8_0.005": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri8"
            self.atol = 0.005
            self.rtol = 0

        if met == "fehlberg2_0.005": #just a quick hack on how to deal with different tolarance values
            self.met = "fehlberg2"
            self.atol = 0.005
            self.rtol = 0
        if met == "bosh3_0.005": #just a quick hack on how to deal with different tolarance values
            self.met = "bosh3"
            self.atol = 0.005
            self.rtol = 0

        if met == "dopri5_0.001": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001
            self.rtol = 0.001
        if met == "dopri5_0.0001": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.0001
            self.rtol = 0.0001
            
        if met == "dopri5_0.001_2": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001
            self.rtol = 0.001
            self.adaptive_step_factor = 2
            self.anode = 4
        if met == "dopri5_0.001_4": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001
            self.rtol = 0.001
            self.adaptive_step_factor = 4
        if met == "dopri5_0.001_5": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001
            self.rtol = 0.001
            self.adaptive_step_factor = 5
            self.anode = 4
        if met == "dopri5_0.002_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 2
            self.rtol =  0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4

        if met == "dopri5_0.002_1.5a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 2
            self.rtol =  0
            self.adaptive_step_factor = 1.5
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "dopri5_0.002_1.2a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 2
            self.rtol =  0
            self.adaptive_step_factor = 1.2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4

        if met == "dopri5_0.005_1.5a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 5
            self.rtol =  0
            self.adaptive_step_factor = 1.5
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "dopri8_0.005_1.5a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri8"
            self.atol = 0.001 * 5
            self.rtol =  0
            self.adaptive_step_factor = 1.5
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "dopri5_0.01_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.01 
            self.rtol =  0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4

        if met == "dopri5_0.0001_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.0001 
            self.rtol =  0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4

        if met == "dopri5_0.005_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 5
            self.rtol =  0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4

        if met == "dopri5_0.001_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 1
            self.rtol =  0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4

        if met == "dopri5_0.01_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 1
            self.rtol =  0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4


        if met == "dopri5_0.002_4a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 2
            self.rtol = 0
            self.adaptive_step_factor = 4
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "dopri5_0.01_4a": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001 * 10
            self.rtol = 0
            self.adaptive_step_factor = 4
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "bosh3_0.005_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "bosh3"
            self.atol = 0.001 * 5
            self.rtol = 0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "bosh3_0.005_1.2a": #just a quick hack on how to deal with different tolarance values
            self.met = "bosh3"
            self.atol = 0.001 * 5
            self.rtol = 0
            self.adaptive_step_factor = 1.2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4

        if met == "fehlberg2_0.005_1.5a": #just a quick hack on how to deal with different tolarance values
            self.met = "fehlberg2"
            self.atol = 0.001 * 5
            self.rtol = 0
            self.adaptive_step_factor = 1.5
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "fehlberg2_0.005_2a": #just a quick hack on how to deal with different tolarance values
            self.met = "fehlberg2"
            self.atol = 0.001 * 5
            self.rtol = 0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.anode = 4
        if met == "rk4_10": #just a quick hack on how to deal with different tolarance values
            self.met = "rk4"
            self.step_size = t[1]/10
        if met == "rk4_20": #just a quick hack on how to deal with different tolarance values
            self.met = "rk4"
            self.step_size = t[1]/20
        if met == "rk4_0.1": 
            self.met = "rk4"
            self.step_size = 0.1
        if met == "rk4_2": 
            self.met = "rk4"
            self.step_size = t[1]/2
        if met == "rk4_1": 
            self.met = "rk4"
            self.step_size = t[1]
        if met == "rk4_5":
            self.met = "rk4"
            self.step_size = t[1]/5

        if met == "euler_2": 
            self.met = "euler"
            self.step_size = t[1]/2
        if met == "euler_5": 
            self.met = "euler"
            self.step_size = t[1]/5
        if met == "euler_4": 
            self.met = "euler"
            self.step_size = t[1]/4
        if met == "euler_10": 
            self.met = "euler"
            self.step_size = t[1]/10

        if met == "mp_2": 
            self.met = "midpoint"
            self.step_size = t[1]/2
        if met == "mp_5": 
            self.met = "midpoint"
            self.step_size = t[1]/5
        if met == "mp_10": 
            self.met = "midpoint"
            self.step_size = t[1]/10

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        if self.anode == 2:
            out = self.odesolver(self.odefunc, x, options = self.options)
            return out
        elif self.anode == 3:
            out,self.n_tot, self.n_acc, self.avg_step, self.ss_loss= self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,steady_state = 0)
            self.ss_loss = (self.ss_loss.flatten(1)).pow(2).sum(1).sqrt().mean()
            return out[1]
        elif self.anode == 4:
            if self.training:
                out,self.liste = self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,adaptive_step_factor = self.adaptive_step_factor, running_mean_step = None) 
                self.running_mean_step.data = torch.clamp((1-0.1)* self.running_mean_step.data + 0.1 * self.liste[-2],0,1)
            else:
                out,self.liste = self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,adaptive_step_factor = self.adaptive_step_factor,running_mean_step = torch.clamp(self.running_mean_step.data,0.0,1.0))
            self.err1 = self.liste[-1]
            self.last_step = self.liste[-2]
            self.states = self.liste[-3]
            self.trajectory_path = self.liste[-4] 
            if self.odefunc.store_activations:
                self.activation_list = self.liste[-5]
                self.liste = self.liste[0:-5]
            else:
                self.liste = self.liste[0:-4]

            return out[1] # Two time points are stored in [0,endtime], here the state at time 1 is output
        elif self.anode == 5:
            out = self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,adaptive_step_factor = self.adaptive_step_factor, traversed_states = self.states) 
            return out
        elif self.anode == 1:
            out = self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol) 
            return out[1]
        else:
            out,self.liste = self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size) 
            return out[1] # Two time points are stored in [0,endtime], here the state at time 1 is output.
    def forward_ss(self,x):
        out,self.n_tot, self.n_acc, self.avg_step, self.ss_loss= self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,steady_state = 0)
        self.ss_loss = (self.ss_loss.flatten(1)).pow(2).sum(1).sqrt().mean()
        return out[1]

    def forward_path(self,x):
        out,self.n_tot, self.n_acc, self.avg_step, self.ss_loss= self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,steady_state = 1)
        self.ss_loss = (self.ss_loss.flatten(1)).pow(2).sum(1).sqrt().mean()
        return out[1]
    
    def set_solver(self,method):
        self.method = method
        if method == 'rk4_10':
            self.met = 'rk4'
            self.step_size= self.integration_time[-1]/10
        elif method == 'rk4_2':
            self.met = 'rk4'
            self.step_size= self.integration_time[-1]/2
        elif method == 'rk4_5':
            self.met = 'rk4'
            self.step_size= self.integration_time[-1]/5
        elif method == 'mp_2':
            self.met = 'midpoint'
            self.step_size= self.integration_time[-1]/2
        elif method == 'mp_5':
            self.met = 'midpoint'
            self.step_size= self.integration_time[-1]/5
        elif method == 'mp_10':
            self.met = 'midpoint'
            self.step_size= self.integration_time[-1]/10
        elif method == 'euler_2':
            self.met = 'euler'
            self.step_size= self.integration_time[-1]/2
        elif method == 'euler_5':
            self.met = 'euler'
            self.step_size= self.integration_time[-1]/5
        elif method == 'euler_10':
            self.met = 'euler'
            self.step_size= self.integration_time[-1]/10
        elif method == 'rk4_0.1':
            self.met = 'rk4'
            self.step_size= 0.1
        elif method == 'dopri5_0.1':
            self.met = 'dopri5'
            self.rtol = 0.1
            self.atol = 0.1
        elif method == 'dopri5':
            self.met = 'dopri5'
            self.rtol = 1e-7
            self.atol = 1e-9
        elif method == 'dopri5_0.001':
            self.met = 'dopri5'
            self.rtol = 0.001
            self.atol = 0.001
        else:
            print("Not implemented yet:",method)
            exit()


class ODEfunc_abstract(nn.Module):
    def __init__(self, in_dim,hid_dim, act = "relu",nogr = 32):
        super(ODEfunc_abstract, self).__init__()
        self.in_dim = in_dim    #I generally allow different numbers of inbetween "channels"
        self.hid_dim = hid_dim

        if act == "relu":
            self.act1 = nn.ReLU(inplace = True)
            self.act2 = nn.ReLU(inplace = True)
        elif act == "elu":
            self.act1 = nn.ELU(inplace=True)
            self.act2 = nn.ELU(inplace=True)
        elif act == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(inplace=True)
        elif act == "sin":
            self.act1 = torch.sin
            self.act2 = torch.sin
        elif act == "cos":
            self.act = torch.cos
        else:
            print("Choosen activation:",act, "not implemented yet.")
            exit()

        self.conv1 = ConcatConv2d(in_dim, hid_dim, 3, 1, 1)
        self.conv2 = ConcatConv2d(hid_dim, in_dim, 3, 1, 1)
        self.nfe = 0
        self.store_activations = False
        self.activations = []

    def forward(self, t, x):
        self.nfe += 1 #count the number of forward passes
        out = self.conv1(t, x)
        if self.store_activations:
            self.activations.append((out > 0).float().mean().item())
        out = self.act1(out)
        out = self.conv2(t, out)
        if self.store_activations:
            self.activations.append((out > 0).float().mean().item())
        out = self.act2(out)
        return out


class ODENet_MNIST(nn.Module):
    def __init__(self,odeblock, device, fixed_layer = 0):
        # odeblock is an instance of ODEBlock
        super(ODENet_MNIST, self).__init__() 
        self.device = device
        self.hid_dim = odeblock.odefunc.in_dim
        normalize_MNIST = [Normalization(device)]
        downsampling_layers = [
            nn.Conv2d(1, self.hid_dim, 3, 1),
            norm(self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_dim, self.hid_dim, 4, 2, 1),
            norm(self.hid_dim),
            nn.ReLU(inplace=True)
            ]
        feature_layers = [odeblock]
        if odeblock.running_mean_step != None:
            odeblock.running_mean_step = odeblock.running_mean_step.to(device)
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(self.hid_dim, 10)   #requires in_dim == out_dim of odeblock !!
            ]
        if fixed_layer == 1:
            fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), Flatten(), MLP_OUT_FIXED(self.hid_dim, 10,device)]

        self.net = nn.Sequential(*normalize_MNIST,*downsampling_layers, *feature_layers, *fc_layers).to(device)

    def forward(self, x):
        return self.net(x)



class ODENet_mini_MNIST(nn.Module):
    def __init__(self,odeblock, device, norm_mean,norm_std,fixed_layer = 0):
        # odeblock is an instance of ODEBlock
        super(ODENet_mini_MNIST, self).__init__() 
        self.device= device
        self.hid_dim = odeblock.odefunc.in_dim
        normalize_MNIST = [Normalization(device,norm_mean,norm_std)]
        downsampling_layers = [
            nn.Conv2d(1, self.hid_dim, 5, 2),
            #norm(self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_dim, self.hid_dim, 5, 2),
            #norm(self.hid_dim),
            nn.ReLU(inplace=True)
            ]
        if odeblock.running_mean_step != None:
            odeblock.running_mean_step.data = odeblock.running_mean_step.data.to(device)
        feature_layers = [odeblock]
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(self.hid_dim, 10)   #requires in_dim == out_dim of odeblock !!
            ]
        if fixed_layer == 1:
            fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), Flatten(), MLP_OUT_FIXED(self.hid_dim, 10,device)]

        self.net = nn.Sequential(*normalize_MNIST,*downsampling_layers, *feature_layers, *fc_layers).to(device)

    def forward(self, x):
        return self.net(x)

class ODENet_mini_CIFAR(nn.Module):
    def __init__(self,odeblock, device, norm_mean,norm_std,fixed_layer = 0):
        # odeblock is an instance of ODEBlock
        super(ODENet_mini_CIFAR, self).__init__() 
        self.device= device
        self.hid_dim = odeblock.odefunc.in_dim
        normalize_MNIST = [Normalization(device,norm_mean,norm_std)]
        downsampling_layers = [
            nn.Conv2d(3, self.hid_dim, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_dim, self.hid_dim, 5, 2),
            nn.ReLU(inplace=True)
            ]
        if odeblock.running_mean_step != None:
            odeblock.running_mean_step.data = odeblock.running_mean_step.data.to(device)
        feature_layers = [odeblock]
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(self.hid_dim, 10)   #requires in_dim == out_dim of odeblock !!
            ]
        if fixed_layer == 1:
            fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), Flatten(), MLP_OUT_FIXED(self.hid_dim, 10,device)]

        self.net = nn.Sequential(*normalize_MNIST,*downsampling_layers, *feature_layers, *fc_layers).to(device)

    def forward(self, x):
        return self.net(x)


class ODENet_mini_bn_MNIST(nn.Module):
    def __init__(self,odeblock, device, norm_mean,norm_std, fixed_layer = 0):
        # odeblock is an instance of ODEBlock
        super(ODENet_mini_bn_MNIST, self).__init__() 
        self.device= device
        self.hid_dim = odeblock.odefunc.in_dim
        normalize_MNIST = [Normalization(device,norm_mean,norm_std)]
        downsampling_layers = [
            nn.Conv2d(1, self.hid_dim, 5, 2),
            nn.BatchNorm2d(self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_dim, self.hid_dim, 5, 2),
            nn.BatchNorm2d(self.hid_dim),
            nn.ReLU(inplace=True)
            ]
        if odeblock.running_mean_step != None:
            odeblock.running_mean_step.data = odeblock.running_mean_step.data.to(device)
        feature_layers = [odeblock]
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(self.hid_dim, 10)   #requires in_dim == out_dim of odeblock !!
            ]
        if fixed_layer == 1:
            fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), Flatten(), MLP_OUT_FIXED(self.hid_dim, 10,device)]

        self.net = nn.Sequential(*normalize_MNIST,*downsampling_layers, *feature_layers, *fc_layers).to(device)

    def forward(self, x):
        return self.net(x)

class ODENet_mini_gn_MNIST(nn.Module):
    def __init__(self,odeblock, device, norm_mean,norm_std,fixed_layer = 0):
        # odeblock is an instance of ODEBlock
        super(ODENet_mini_gn_MNIST, self).__init__() 
        self.device= device
        self.hid_dim = odeblock.odefunc.in_dim
        normalize_MNIST = [Normalization(device,norm_mean,norm_std)]
        downsampling_layers = [
            nn.Conv2d(1, self.hid_dim, 5, 2),
            norm(self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_dim, self.hid_dim, 5, 2),
            norm(self.hid_dim),
            nn.ReLU(inplace=True)
            ]
        if odeblock.running_mean_step != None:
            odeblock.running_mean_step.data = odeblock.running_mean_step.data.to(device)
        feature_layers = [odeblock]
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(self.hid_dim, 10)   #requires in_dim == out_dim of odeblock !!
            ]
        if fixed_layer == 1:
            fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), Flatten(), MLP_OUT_FIXED(self.hid_dim, 10,device)]

        self.net = nn.Sequential(*normalize_MNIST,*downsampling_layers, *feature_layers, *fc_layers).to(device)

    def forward(self, x):
        return self.net(x)


class ODEfunc_bn(nn.Module):
    def __init__(self, in_dim,hid_dim,device,act = "relu",n_max_tp = 31,endtime = 1.0):
        super(ODEfunc_bn, self).__init__()
        self.in_dim = in_dim    #I generally allow different numbers of inbetween "channels"
        self.hid_dim = hid_dim

        if act == "relu":
            self.act1 = nn.ReLU(inplace = True)
            self.act2 = nn.ReLU(inplace = True)
        elif act == "sin":
            self.act1 = torch.sin
            self.act2 = torch.sin
        elif act == "cos":
            self.act = torch.cos
        else:
            print("Choosen activation:",act, "not implemented yet.")
            exit()

        self.conv1 = ConcatConv2d(in_dim, hid_dim, 3, 1, 1)
        self.conv2 = ConcatConv2d(hid_dim, in_dim, 3, 1, 1)

        self.norm1 = BatchNorm(hid_dim,device,n_max_tp, endtime) # all timepoints that that apprear in rk4_10
        self.norm2 = BatchNorm(in_dim,device,n_max_tp, endtime)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1 #count the number of forward passes
        out = self.conv1(t, x)
        out = self.norm1(t,out)
        out = self.act1(out)
        out = self.conv2(t, out)
        out = self.norm2(t,out)
        out = self.act2(out)
        return out

