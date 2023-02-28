import numpy as np
import os
import time
import argparse
import pandas as pd


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
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--adv', default="", type=str,
                    help='indicates whether "" normal training or "CW" adv training')

parser.add_argument('--traj_mode', default = 0, type = int)
parser.add_argument('--target', default = 40, type = float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--omega', default=0.0, type=float)
parser.add_argument('--zono', action='store_true')
parser.add_argument('--fixed_traj', action='store_true')
parser.add_argument('--ablation', action='store_true')
parser.add_argument('--q_start', default=0.15, type=float)
parser.add_argument('--q_end', default=0.33, type=float)
parser.add_argument('--ptm', default=0, type=int, help="provable training mode")
parser.add_argument('--kappa', default=0, type=int, help="provable training mode")
parser.add_argument('--pre_trained',action = 'store_true')

args = parser.parse_args()
if args.gpu < 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)


import torch 
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from model import *
import pdb
import torchvision
import torchattacks

from PARC.AIDomains.deeppoly import DeepPoly, backprop_dp, backward_deeppoly, compute_dp_relu_bounds,forward_deeppoly



def contained(zono,x):
    lb,ub = zono.concretize()

    check = ((lb > x).sum() == 0) and ((ub < x).sum() == 0)
    return check.item()


def main():
    if args.gpu < 3:
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu')


    from PARC.AIDomains.zonotope import HybridZonotope
    from PARC.AIDomains.abstract_layers import Conv2d, Linear, ReLU, GlobalAvgPool2d, Flatten, BatchNorm2d, Upsample, Log, Exp, \
        Inv, LogSumExp, Entropy, BatchNorm1d, AvgPool2d, Bias, Scale, Normalization, BasicBlock, WideBlock, FixupBasicBlock, Sequential, TestBlock, GroupNorm, ConcatConv, ODEBlock_A


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

    if args.ablation:
        resume = "{0}seed_{1}_target_{2}_qstart_{3}_qend_{4}_ptm_{5}_kappa_{6}.pth.tar".format(args.adv,args.seed,args.target,args.q_start,args.q_end,args.ptm,args.kappa)
    
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

    if args.dataset == "MNIST":
        epsilons = [0.1,0.15,0.2]
        verified_lists = [[1]*args.samples for _ in range(len(epsilons) + 1)]
        index_dict = {0:[1,2,3],1:[2,3],2:[3],3:[]}

    elif args.dataset == "F-MNIST":
        epsilons = [0.1,0.15]
        verified_lists = [[1]*args.samples for _ in range(len(epsilons) + 1)]
        index_dict = {0:[1,2],1:[2],2:[]}

    elif args.dataset == "CIFAR10":
        epsilons = [1 / 255]
        verified_lists = [[1]*args.samples for _ in range(len(epsilons) + 1)]
        index_dict = {0:[1],1:[]}

    try:
        exists_prelim = True
        resume = args.exppath + "prelim_results/{0}_seed_{1}_target_{2}.tar.pth".format(args.adv,args.seed,args.target)
        prelim_list = torch.load(resume)["list"]
    except:
        #print("no prelim")
        exists_prelim = False
        prelim_list = [[[],[]] for x in range(len(verified_lists))]

    sim_min = []
    sim_mean = []
    c_l,d_l,a_l = [],[],[]

    for incorrect in prelim_list[0][1]:
        verified_lists[0][incorrect] = 0

    for jjj in range(len(epsilons)):
        eps =  epsilons[jjj]
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader),  total=args.samples):
                
                if batch_idx == args.samples:
                    print(np.sum(verified_lists[jjj+1]))
                    break

                if batch_idx in prelim_list[jjj+1][0]:
                    #this sample has already been certified with this eps
                    continue

                if batch_idx in prelim_list[jjj+1][1]:
                    #adversarial example found for this eps
                    verified_lists[jjj+1][batch_idx] = 0
                    continue

                if verified_lists[jjj+1][batch_idx] == 0:
                    continue

                inputs = data.type(torch.DoubleTensor).to(device)
                target = target.to(device)
                if not exists_prelim:
                
                    out = model(inputs)
                    if not torch.argmax(out) == target:
                        verified_lists[0][batch_idx] = 0
                        temp_list = index_dict[0]
                        for indexx in temp_list:
                            verified_lists[indexx][batch_idx] = 0
                        continue


                #########################
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

                    #min area heuristic
                    dp = DeepPoly(expr_coef = expr_coef)
                    #lambda = 0 slopes
                    dp_0 = DeepPoly(expr_coef = expr_coef)
                    #lambda = 1 slopes
                    dp_1 = DeepPoly(expr_coef = expr_coef)
                    for i in range(j-1,-1,-1):
                        dp = backprop_dp(net.layers[i], dp, it= 10, use_lambda=False,time = 0.0)
                        dp_0 = backprop_dp(net.layers[i], dp_0, it= 17, use_lambda=False,time = 0.0)
                        dp_1 = backprop_dp(net.layers[i], dp_1, it= 18, use_lambda=False,time = 0.0)
                    if j == idx[-1]:
                        low_dp,upp_dp = dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
                        low_dp_0,upp_dp_0 = dp_0.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
                        low_dp_1,upp_dp_1 = dp_1.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))

                        low_dp = torch.maximum(low_dp_0,low_dp)
                        upp_dp = torch.minimum(upp_dp_0,upp_dp)
                        low_dp = torch.maximum(low_dp_1,low_dp)
                        upp_dp = torch.minimum(upp_dp_1,upp_dp)
                    else:
                        low_dp,upp_dp = dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
                        low_dp_0,upp_dp_0 = dp_0.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
                        low_dp_1,upp_dp_1 = dp_1.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))

                        low_dp_comb = torch.maximum(low_dp_0,low_dp)
                        upp_dp_comb = torch.minimum(upp_dp_0,upp_dp)
                        low_dp_comb = torch.maximum(low_dp_1,low_dp_comb)
                        upp_dp_comb = torch.minimum(upp_dp_1,upp_dp_comb)

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
                    if isinstance(net.layers[i],ODEBlock_A):
                        break
                    dp= backprop_dp(net.layers[i], dp, it= 10, use_lambda=False,time = 0.0)

                lb_hyb,ub_hyb = dp.dp_concretize(bounds = (input_box.concretize()))

                lb = torch.maximum(lb_hyb,torch.maximum(low_box,low_dp))
                

                
                #we need to check if low >0
                if not (lb > 0).all():
                    verified_lists[jjj+1][batch_idx] = 0
                    temp_list = index_dict[jjj+1]
                    for indexx in temp_list:
                        verified_lists[indexx][batch_idx] = 0
            
    print("dataset,target,omega,seed",args.dataset,args.target, args.omega,args.seed)

    for i in range(len(verified_lists)):
        if i == 0:
            print("Standard ACC [%]:",np.mean(verified_lists[i])*100)
        else:
            print("Certified ACC [%] with eps = {0}:".format(epsilons[i-1]),np.mean(verified_lists[i])*100)

    end = time.time()

    res1 = args.exppath + "gains_results/"
    if not os.path.isdir(res1):
        os.makedirs(res1)
    resume = args.exppath + "gains_results/{0}_seed_{1}_target_{2}.tar.pth".format(args.adv,args.seed,args.target)
    if args.ablation:
        resume = args.exppath + "gains_results/box_seed_{0}_target_{1}_qstart_{2}_qend_{3}_ptm_{4}_kappa_{6}.tar.pth".format(args.seed,args.target,args.q_start,args.q_end,args.ptm,args.kappa)

    torch.save({"list":verified_lists},resume)


if __name__ == '__main__':
    main()



























