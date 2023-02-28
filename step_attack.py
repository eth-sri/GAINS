import numpy as np
import os
import time
import copy
import argparse

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--act', type = str, default = "mini")
parser.add_argument('--exppath', type = str, default = "models/")
parser.add_argument('--method', default="dopri5_0.005_2a", type=str)
parser.add_argument('--endtime', default=1.0, type=float)
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--adv', default="", type=str,
                    help='indicates whether "" normal training or "CW" adv training')
parser.add_argument('--adjoint', default=0, type=int)
parser.add_argument('--odeint', default="DTO", type=str)
parser.add_argument('--nogr', default=32, type=int)
parser.add_argument('--psi', default=1.0, type=float)
parser.add_argument('--eps', default=1.0, type=float)
parser.add_argument('--test', default=0, type=int)
parser.add_argument('--ss_loss', default=0.1, type=float)
parser.add_argument('--onlytime', default=0, type=int)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--schedule', default = 5, type = int)
parser.add_argument('--dataset', type = str, default = "MNIST")

parser.add_argument('--debug', default = 0, type = int)
parser.add_argument('--traj_mode', default = 0, type = int)
parser.add_argument('--target', default = 40.0, type = float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--delta', default=0.66, type=float)
parser.add_argument('--omega', default=0.0, type=float)
parser.add_argument('--pre_trained',action = 'store_true')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

import pandas as pd
import torch as t
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from model import *
import pdb
import torchvision
import torchattacks
from utils import *

def update_statistics(stats,liste):
    if len(stats) == 0:
        stats = [[] for i in range(len(liste))]
    for i in range(len(liste)):
        stats[i] += [liste[i]]
    return stats

def main():    
    device = t.device('cuda:0') 

    if args.seed != 0:
        t.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.pre_trained:
        args.exppath = args.exppath + "pre-trained/"
    else:
        args.exppath = args.exppath + "trained/"
    args.exppath = args.exppath + args.dataset + "/"
    n_test = args.batch_size

    if args.dataset == "MNIST":
        #MNIST DATASET
        norm_mean,norm_std = [0.1307],[0.3081]
        test_loader = t.utils.data.DataLoader(
                torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=args.batch_size, shuffle=False)

    elif args.dataset == "F-MNIST":
        #FMNIST DATASET
        norm_mean,norm_std = [0.2860], [0.3530]
        test_loader = t.utils.data.DataLoader(
                torchvision.datasets.FashionMNIST("data/f-mnist", train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=args.batch_size, shuffle=False)

    #get desired number of samples
    first_step_list = [False for x in range(args.batch_size)]
    trajectory_list = [False for x in range(args.batch_size)]
    for batch_idx, (data, target) in tqdm(enumerate(test_loader),  total=len(test_loader)):
        images = data
        targets = target
        break
    odefunc = ODEfunc_abstract(32,32,act = "relu",nogr = args.nogr)
    ODE = ODEBlock(odefunc,args.method, 0,[0,args.endtime])
    model = ODENet_mini_MNIST(ODE,device,norm_mean,norm_std)   

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    if args.adv in ["PGD",""]:
        resume = args.exppath + "{2}activation_{0}_method_{1}_checkpoint_epoch_{3}_seed_{4}.pth.tar".format(args.act,args.method,args.adv,args.epochs,args.seed)
    else:
        resume =  args.exppath  + "{2}activation_{0}_method_{1}_epoch_{3}_seed_{4}_target_{5}_gamma_{6}_omega_{7}_traj_{8}_run_{9}.pth.tar".format(args.act,args.method,args.adv,args.epochs,args.seed,args.target,args.gamma,args.omega,args.traj_mode,args.psi)

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = t.load(resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1'] 
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        exit()
    if not t.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        model = model.to(device)
    model.eval()

    sd = model.state_dict()
    current_list = []
    accuracies = []
    accuracies.append("{0} {1} ({2})".format(args.act,args.method,args.endtime))
    
    odefunc_test = ODEfunc_abstract(32,32,act = "relu",nogr = args.nogr)
    ODE_test = ODEBlock(odefunc_test,args.method, 0,[0,args.endtime])
    test_model = ODENet_mini_MNIST(ODE_test,device,norm_mean,norm_std)
    test_model.load_state_dict(sd)
    test_model.eval()
    boolean = args.act not in ["baseline","baseline_sin","baseline_sin_10_ps","baseline_sin_10_ns","baseline_10_ps","baseline_10_ns"]
    boolean_value = 7
    if args.act == "mini":
        boolean_value = 5
    args.batch_size = 1
    attackers = []

    attackers.append(torchattacks.PGD(model, eps= 0.1,steps = 100,alpha = 0.1 / 40))
    attackers.append(torchattacks.PGD(model, eps= 0.15,steps = 100,alpha = 0.1 / 40)) 
    attackers.append(torchattacks.PGD(model, eps= 0.2,steps = 100,alpha = 0.1 / 40))
    if args.adv == "box":
        accuracies = ["{0} {1} {2} {3} ".format(args.adv,args.target,args.omega,args.psi)]
    else:
        accuracies = ["{0}".format(args.adv,args.target,args.omega,args.psi)]

    for adv in attackers:
        attack_success = 0
        attack_traj = 0
        possible_attacks = 0
        pot_2 = 0
        pot_2_success = 0
        test_acc = 0.0
        avg_step = []
        test_list = []
        l2_dist = []

        for i in range(int(images.shape[0]//args.batch_size)):
            data = images[i*args.batch_size: (1+i)*args.batch_size].clone().to(device)
            target = targets[i*args.batch_size: (1+i)*args.batch_size].clone().to(device)
            out = model(data)
            trajectory = model.net[5].trajectory_path
            err1,err2 = model.net[boolean_value].err1
            flagge = True
            if trajectory_list[i]:
                attack_traj += 1
                possible_attacks += 1
                if first_step_list[i]:
                    attack_success += 1
                continue
            if err1 <= 2 ** (-5): # mode 0, increase error
                action = 2
                mode = [0]
                save = 0
            elif err1 > 1: #mode 1 decrease error
                mode = [1]
                action = 0
                save = 1
            else:
                action = 1
                mode = [1]
                save = 1
                if (1-err1) < (err1 - 2 ** (-5)):
                    save= 0
                    mode = [0]
            if err2 <= 2 ** (-5): # mode 0, increase error
                action2 = 2
                mode2 = [0]
            elif err2 > 1: #mode 1 decrease error
                mode2 = [1]
                action2 = 0
            else:
                action2 = 1
                mode2 = [1]
                if (1-err2) < (err2 - 2 ** (-5)):
                    mode2 = [0]
            for schedule in [5,4,3,2,1,0,-1]:
                if not flagge:
                    break
                args.schedule = schedule
                for ii in range(len(mode)):
                    if args.schedule == 0:
                        adv_images = adv(data, err1.clone(), mode[ii])
                    else:
                        adv_images = adv(data, err1.clone(), mode[ii],mode2[ii],args.schedule)
                    with t.no_grad(): 
                        output = test_model(adv_images)
                        attacked_trajectory = test_model.net[5].trajectory_path
                        err3,err4 = test_model.net[boolean_value].err1
                        if err3 <= 2 ** (-5):
                            ref_action = 2
                        elif err3 >1:
                            ref_action = 0
                        else:
                            ref_action = 1
                        if err4 <= 2 ** (-5):
                            ref_action2 = 2
                        elif err4 >1:
                            ref_action2 = 0
                        else:
                            ref_action2 = 1
                        if ((trajectory != attacked_trajectory) or (schedule == -1)) and boolean:
                            attack_traj += (trajectory != attacked_trajectory)
                            trajectory_list[i] = (trajectory != attacked_trajectory)
                            first_step_list[i] = (ref_action != action)
                            possible_attacks += 1
                            attack_success += (ref_action != action)
                            flagge = False
                            test_acc += t.sum(t.argmax(output, dim=1) == target).item()
        accuracies.append(attack_traj)
        print("EPS:",adv.eps)
        print("TRAJECTORY attack Success [%]:",(attack_traj/possible_attacks)*100)

    current_list.append(np.array(accuracies))
    res1 = args.exppath + "traj_results/"
    if not os.path.isdir(res1):
        os.makedirs(res1)
    resume = args.exppath + "traj_results/{0}_seed_{1}_target_{2}.tar.pth".format(args.adv,args.seed,args.target)
    torch.save({"list":accuracies},resume)
    end_time = time.time()
    mins = (end_time - start_time )//60
    secs = round((end_time - start_time )%60,0)
    print("Time:{0} min {1} sec".format(mins,secs))

if __name__ == '__main__':
    main()



























