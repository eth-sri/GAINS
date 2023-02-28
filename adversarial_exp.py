import numpy as np
import os
import time

import argparse

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
parser.add_argument('--fixed_traj', action='store_true')
parser.add_argument('--pre_trained',action = 'store_true')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

import pandas as pd
import torch 
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from model import *
import pdb
import torchvision
import torchattacks


from PARC.AIDomains.deeppoly import DeepPoly, backprop_dp, backward_deeppoly, compute_dp_relu_bounds

def main():
    device = torch.device('cuda:0') 
    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
        #CIFAR DATASET
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

    attackers = []
    if args.dataset == "MNIST":
        attackers.append(torchattacks.PGD(model, eps= 0.1,steps = 200,alpha = 0.1 / 40))
        attackers.append(torchattacks.PGD(model, eps= 0.15,steps = 200,alpha = 0.1 / 40)) 
        attackers.append(torchattacks.PGD(model, eps= 0.2,steps = 200,alpha = 0.1 / 40))

        adversarial_lists = [[1]*args.samples for _ in range(len(attackers) + 1)]
        index_dict = {0:[1,2,3],1:[2,3],2:[3],3:[]}

    elif args.dataset == "F-MNIST":
        attackers.append(torchattacks.PGD(model, eps= 0.1,steps = 200,alpha = 0.1 / 40))
        attackers.append(torchattacks.PGD(model, eps= 0.15,steps = 200,alpha = 0.1 / 40)) 
        adversarial_lists = [[1]*args.samples for _ in range(len(attackers) + 1)]
        index_dict = {0:[1,2],1:[2],2:[]}

    elif args.dataset == "CIFAR10":
        attackers.append(torchattacks.PGD(model, eps= 0.001,steps = 200,alpha = 1 / (255*40)))
        adversarial_lists = [[1]*args.samples for _ in range(len(attackers) + 1)]
        index_dict = {0:[1],1:[]}


    traj_list_count = [0,0,0,0]
    for jjj in range(len(attackers)):
        adv = attackers[jjj]

        for batch_idx, (data, target) in tqdm(enumerate(test_loader),  total=args.samples):
            if batch_idx == args.samples:
                print(np.sum(adversarial_lists[jjj+1]))
                break
            if adversarial_lists[jjj+1][batch_idx] == 0:
                continue

            inputs = data.to(device)
            target = target.to(device)
            if args.fixed_traj:
                model.net[5].anode = 4

            out = model(inputs) #get reference path
            traj = model.net[5].trajectory_path 
            if not torch.argmax(out) == target:
                adversarial_lists[0][batch_idx] = 0
                temp_list = index_dict[0]
                for indexx in temp_list:
                    adversarial_lists[indexx][batch_idx] = 0
                continue
            if args.fixed_traj:
                model.net[5].anode = 5

            adv_inputs = adv(data, target)
            adv_out =  model(adv_inputs)

            if args.fixed_traj:
                model.net[5].anode = 4
            out_2 = model(adv_inputs)
            
            counter = int(traj != model.net[5].trajectory_path)
            if not torch.argmax(adv_out) == target:
                adversarial_lists[jjj+1][batch_idx] = 0
                traj_list_count[jjj+1] += counter
                temp_list = index_dict[jjj+1]
                for indexx in temp_list:
                    adversarial_lists[indexx][batch_idx] = 0
                    traj_list_count[indexx] += counter

            
    print("dataset,target,omega,seed",args.dataset,args.target, args.omega,args.seed)

    for i in range(len(adversarial_lists)):
        if i == 0:
            print("Standard ACC [%]:",np.mean(adversarial_lists[i])*100)
        else:
            print("Adv. ACC [%] with eps = {0} :".format(attackers[i-1].eps),np.mean(adversarial_lists[i])*100)

    res1 = args.exppath + "adversarial_results/"
    if not os.path.isdir(res1):
        os.makedirs(res1)
    resume = args.exppath + "adversarial_results/{0}_seed_{1}_target_{2}.tar.pth".format(args.adv,args.seed,args.target)
    torch.save({"list":adversarial_lists},resume)


if __name__ == '__main__':
    main()



























