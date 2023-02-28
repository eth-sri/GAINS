import numpy as np
import os
import time

import pandas as pd
import torch 
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from model import *
import argparse
import pdb
import torchvision
import torchattacks

from PARC.AIDomains.deeppoly import DeepPoly, backprop_dp, backward_deeppoly, compute_dp_relu_bounds
from PARC.AIDomains.zonotope import HybridZonotope
from PARC.AIDomains.abstract_layers import Conv2d, Linear, ReLU, GlobalAvgPool2d, Flatten, BatchNorm2d, Upsample, Log, Exp, \
    Inv, LogSumExp, Entropy, BatchNorm1d, AvgPool2d, Bias, Scale, Normalization, BasicBlock, WideBlock, FixupBasicBlock, Sequential, TestBlock, GroupNorm, ConcatConv, ODEBlock_A

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--eps',  default=0.001, type=float)
parser.add_argument('--psi',  default=1.0, type=float)
parser.add_argument('--act', type = str, default = "mini")
parser.add_argument('--exppath', type = str, default = "models/")
parser.add_argument('--method', default="dopri5_0.005_2a", type=str)
parser.add_argument('--endtime', default=1.0, type=float)
parser.add_argument('--samples', default=1000, type=int)
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--dataset', type = str, default = "MNIST")
parser.add_argument('--mode', type = str, default = "GAINS")
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--adv', default="", type=str,
                    help='indicates whether "" normal training or "CW" adv training')

parser.add_argument('--traj_mode', default = 0, type = int)
parser.add_argument('--target', default = 40, type = float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--omega', default=0.0, type=float)
parser.add_argument('--fixed_traj', action='store_true')
parser.add_argument('--titan',action = 'store_true')
parser.add_argument('--pre_trained',action = 'store_true')


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)
    #device = torch.device('cuda:0') 
    device = torch.device('cpu')

    if args.mode not in ["PGD","GAINS","GAINS-Box", "GAINS-Linear"]:
        print("Not valid mode:",args.mode)
        exit()

    if args.pre_trained:
        args.exppath = args.exppath + "pre-trained/" 

    args.exppath = args.exppath + args.dataset + "/"

    if args.dataset == "MNIST":
        norm_mean,norm_std = [0.1307],[0.3081]
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                transform=torchvision.transforms.ToTensor()), batch_size=args.batch_size, shuffle=False)
    elif args.dataset == "F-MNIST":
        #FMNIST DATASET
        norm_mean,norm_std = [0.2860], [0.3530]
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST("data/f-mnist", train=False, download=True,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=1, shuffle=False)
    elif args.dataset == "CIFAR10":
        #FMNIST DATASET
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2023, 0.1994, 0.2010]
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10("data/cifar10",train = False,download=True,transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()])),batch_size= 1,shuffle = False)

    if args.dataset in ["CIFAR10"]:
        dimss = 32
        odefunc = ODEfunc_abstract(dimss,dimss,act = "relu",nogr = 32)
        ODE = ODEBlock(odefunc,args.method, 0,[0,args.endtime])
        model = ODENet_mini_CIFAR(ODE,device,norm_mean,norm_std)
    else:
        odefunc = ODEfunc_abstract(32,32,act = "relu",nogr = 32)
        ODE = ODEBlock(odefunc,args.method, 0,[0,args.endtime])
        model = ODENet_mini_MNIST(ODE,device,norm_mean,norm_std)

    if args.adv in ["PGD",""]:
        resume = args.exppath + "{2}activation_{0}_method_{1}_checkpoint_epoch_{3}_seed_{4}.pth.tar".format(args.act,args.method,args.adv,args.epochs,args.seed)
    else:
        resume =  args.exppath  + "{2}activation_{0}_method_{1}_epoch_{3}_seed_{4}_target_{5}_gamma_{6}_omega_{7}_traj_{8}_run_{9}.pth.tar".format(args.act,args.method,args.adv,args.epochs,args.seed,args.target,args.gamma,args.omega,args.traj_mode,args.psi)

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume,map_location = device)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1'] 
        if "net.5.running_mean_step" in model.state_dict().keys() and "net.5.running_mean_step" not in checkpoint['state_dict'].keys():
            checkpoint['state_dict']["net.5.running_mean_step"] = nn.parameter.Parameter(torch.ones(1)*0.2,requires_grad= False)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        exit()


    model = model.to(device)
    model.eval()
    if args.mode == "PGD":
        attackers = []
        attackers.append(torchattacks.PGD(model, eps= args.eps,steps = 300,alpha = args.eps / 40))
    else:
        model.double()

    #prepare abstract model
    if args.dataset == "CIFAR10":
        net = Sequential.from_concrete_network(model.net,(3,32,32)) 
    else:
        net = Sequential.from_concrete_network(model.net,(1,28,28)) 
    if args.fixed_traj:
        net[5].mode = "abstract_fixed"
    else:
        net[5].mode = "abstract"
    idx = []
    for i in range(len(net.layers)):
        if net.layers[i].needs_bounds:
            if isinstance(net.layers[i],ODEBlock_A) and (i-1) not in idx:
                idx.append(i-1)
            idx.append(i)

    if len(net.layers) not in idx:
        idx.append(len(net.layers))

    final_lin_layers = []
    for jj in range(10):
        #get final layer of verification depending on target
        eye = (-1)*torch.eye(10).to(device)
        eye[jj,:] += 1
        final_lin_mat = torch.cat((eye[:,0:jj],eye[:,jj+1:]),dim = 1).double()
        final_lin_layer =  nn.Linear(10,9,bias = None)
        final_lin_layer.weight.data = final_lin_mat.T
        final_lin_layer.double().to(device)
        final_lin_layers.append(final_lin_layer)
    temp_dict = {}
    for ii in idx:
        temp = 1
        if ii == 0:
            for x in net.layers[0].output_dim:
                temp *= x
        else:
            for x in net.layers[ii-1].output_dim:
                temp *= x
        temp_dict[ii] = temp
    start = time.time()
    epsilons = [args.eps]
    eps = args.eps

    sim_min = []
    sim_mean = []
    c_l,d_l,a_l = [],[],[]

    differences = []


    for jjj in range(len(epsilons)):
        eps =  epsilons[jjj]
        for batch_idx, (data, target) in tqdm(enumerate(test_loader),  total=args.samples):
            if batch_idx == args.samples:
                break


            inputs = data.type(torch.DoubleTensor).to(device)
            target = target.to(device)

            if args.mode == "GAINS":
                with torch.no_grad():
                    lb= gains_bounds(inputs,target,eps,net, final_lin_layers,idx,temp_dict,device)
            elif args.mode == "GAINS-Box":
                with torch.no_grad():
                    lb= gains_box_bounds(inputs,target,eps,net, final_lin_layers,idx,temp_dict,device)
            elif args.mode == "GAINS-Linear":
                with torch.no_grad():
                    lb= gains_linear_bounds(inputs,target,eps,net, final_lin_layers,idx,temp_dict,device)
            elif args.mode == "PGD":
                lb= PGD_bounds(model,data,target,attackers[jjj])

            differences.append(lb.min().item())

                                

    end = time.time()


    res1 = args.exppath + "bounds/"
    if not os.path.isdir(res1):
        os.makedirs(res1)
    resume = args.exppath + "bounds/{0}_seed_{1}_target_{2}_eps_{3}_bounds_{4}.tar.pth".format(args.adv,args.seed,args.target,args.eps,args.mode)
    torch.save({"bounds":differences},resume)

def PGD_bounds(model,data,target,adv):
    adv_inputs = adv(data, target)
    adv_out =  model(adv_inputs)
    temp = adv_out[0,target] - adv_out
    abc = temp[torch.where((temp > 1e-4).float() + (temp < -1e-4).float())]
    if abc.shape[0] == 0:
        abc = temp
    return abc

def gains_box_bounds(inputs,target,eps,net, final_lin_layers,idx,temp_dict,device):
    input_box = HybridZonotope.construct_from_noise(x=inputs.clone(),dtype = inputs.dtype, eps=eps, domain="box", data_range=(0.0, 1.0))

    abstract_output = net.forward_between(0,len(net),input_box.clone())
    abstract_output = abstract_output.linear(final_lin_layers[target[0]].weight.data,None)

    lb,ub = abstract_output.concretize()
    return lb
def gains_linear_bounds(inputs,target,eps,net, final_lin_layers,idx,temp_dict,device):
    input_low, input_upp = torch.clamp(inputs.clone()-eps,0.0,1.0) ,torch.clamp(inputs.clone()+eps,0.0,1.0) 
    for j in idx:
        if j == len(net.layers):
            expr_coef = final_lin_layers[target[0]].weight.data
        elif j == 0:
            expr_coef = torch.eye(temp_dict[j]).view(-1, *net.layers[0].output_dim).unsqueeze(0).type(torch.DoubleTensor).to(device)
        else:
            expr_coef = torch.eye(temp_dict[j]).view(-1, *net.layers[j-1].output_dim).unsqueeze(0).type(torch.DoubleTensor).to(device)

        dp = DeepPoly(expr_coef = expr_coef)
        for i in range(j-1,-1,-1):
            dp= backprop_dp(net.layers[i], dp, it= 10, use_lambda=False,time = 0.0)
        if j == len(net.layers):
            low,upp = dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
        else:
            net.layers[j].update_bounds(dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone())))
            if isinstance(net.layers[j],ODEBlock_A):

                box = HybridZonotope.construct_from_bounds(min_x= net.layers[j].bounds[0], 
                                max_x=net.layers[j].bounds[1], 
                                dtype = net.layers[j].bounds[0].dtype,
                                domain = "box") 
                abc = net.layers[j].forward(box)

    low = low.view(-1,9)
    upp = upp.view(-1,9)
    return low

def gains_bounds(inputs,target,eps,net, final_lin_layers,idx,temp_dict,device):
    prev = 0
    input_low, input_upp = torch.clamp(inputs.clone()-eps,0.0,1.0) ,torch.clamp(inputs.clone()+eps,0.0,1.0) 
    input_box = HybridZonotope.construct_from_noise(x=inputs.clone(),dtype = inputs.dtype, eps=eps, domain="box", data_range=(0.0, 1.0))
    
    for j in idx:
        if j == len(net.layers):
            expr_coef = final_lin_layers[target[0]].weight.data
        elif j == 0:
            expr_coef = torch.eye(temp_dict[j]).view(-1, *net.layers[0].output_dim).unsqueeze(0).type(torch.DoubleTensor).to(device)
        else:
            expr_coef = torch.eye(temp_dict[j]).view(-1, *net.layers[j-1].output_dim).unsqueeze(0).type(torch.DoubleTensor).to(device)

        dp = DeepPoly(expr_coef = expr_coef)
        dp_0 = DeepPoly(expr_coef = expr_coef)
        for i in range(j-1,-1,-1):
            dp = backprop_dp(net.layers[i], dp, it= 10, use_lambda=False,time = 0.0)
            dp_0 = backprop_dp(net.layers[i], dp_0, it= 17, use_lambda=False,time = 0.0)
        if j == idx[-1]:
            low_dp,upp_dp = dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
            low_dp_0,upp_dp_0 = dp_0.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))

            low_dp = torch.maximum(low_dp_0,low_dp)
            upp_dp = torch.minimum(upp_dp_0,upp_dp)
        else:
            low_dp,upp_dp = dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
            low_dp_0,upp_dp_0 = dp_0.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
            low_dp_comb = torch.maximum(low_dp_0,low_dp)
            upp_dp_comb = torch.minimum(upp_dp_0,upp_dp)

            prev_output = net.forward_between(prev,j,input_box)

            low_box, upp_box = prev_output.concretize()
            tmp = low_box.shape
            lb = torch.maximum(low_box,low_dp_comb.reshape(tmp))
            ub = torch.minimum(upp_box,upp_dp_comb.reshape(tmp))
            net.layers[j].update_bounds((lb,ub))
            prev = j 
            input_box = HybridZonotope.construct_from_bounds(min_x= net.layers[j].bounds[0], 
                                max_x=net.layers[j].bounds[1], 
                                dtype = net.layers[j].bounds[0].dtype,
                                domain = "box") 

            if isinstance(net.layers[j],ODEBlock_A):
                #just such that we have graph
                input_box = net.layers[j].forward(input_box)
                prev = j+1

    #########################

    abstract_output = net.forward_between(prev,len(net),input_box.clone())
    abstract_output = abstract_output.linear(final_lin_layers[target[0]].weight.data,None)

    low_box,upp_box = abstract_output.concretize()

    
    expr_coef = final_lin_layers[target[0]].weight.data
    dp = DeepPoly(expr_coef = expr_coef)
    for i in range(len(net)-1,-1,-1):
        #print(net.layers[i])
        if isinstance(net.layers[i],ODEBlock_A):
            break
        dp= backprop_dp(net.layers[i], dp, it= 10, use_lambda=False,time = 0.0)

    
    lb_hyb,ub_hyb = dp.dp_concretize(bounds = (input_box.concretize()))

    lb = torch.maximum(lb_hyb,torch.maximum(low_box,low_dp))
    return lb

if __name__ == '__main__':
    main()



























