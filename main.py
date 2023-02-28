import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=512)
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
parser.add_argument('--accumulate', default=1, type=int)
parser.add_argument('--nogr', default=32, type=int)
parser.add_argument('--ss_loss', default=1, type=float)

parser.add_argument('--lamda', default=1.0, type=float)
parser.add_argument('--psi', default=1.0, type=float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--delta', default=0.66, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--omega', default=0.0, type=float)
parser.add_argument('--clip', default=1000.0, type=float)
parser.add_argument('--q_start', default=0.15, type=float)
parser.add_argument('--q_end', default=0.33, type=float)
parser.add_argument('--ptm', default=0, type=int, help="provable training mode")
parser.add_argument('--kappa', default=0, type=int, help="provable training mode")

parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--debug', default = 0, type = int)
parser.add_argument('--dyn', default = 1, type = int)
parser.add_argument('--start', default = 0, type = int)
parser.add_argument('--target', default = 1.5, type = float)
parser.add_argument('--dataset', type = str, default = "MNIST")
parser.add_argument('--traj_mode', default = 0, type = int)
parser.add_argument('--cold_start', action='store_true', help="Include extrapolated points")
parser.add_argument('--no_warm_up', action='store_true', help="Include extrapolated points")
parser.add_argument('--fc', action = "store_true")
parser.add_argument('--full', action = "store_true")
parser.add_argument('--final', action = "store_true")

#FIRST set device before loading torch
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

import torch as t
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from model import *

import pdb
import torchvision
import torchattacks
from torch.utils.tensorboard import SummaryWriter

from PARC.AIDomains.zonotope import HybridZonotope
from PARC.AIDomains.abstract_layers import Sequential,Linear
from PARC.examples.eps_scheduler import SmoothedScheduler
from utils import save_checkpoint,check_sound,test,update_statistics,train,train_adv, train_box_provable,load_dataset

def main():

    device = t.device('cuda:0') 

    writer = SummaryWriter(args.exppath+ '/runs/{8}_{2}activation_{0}_method_{1}_target_{3}_gamma_{4}_omega_{5}_traj_{6}_run_{7}'.format(args.act,args.method,args.adv,args.target,args.gamma,args.omega,args.traj_mode,args.psi,args.dataset)) 
    
    args.exppath = args.exppath + args.dataset + "/"
    if not os.path.isdir(args.exppath):
        os.makedirs(args.exppath)

    if args.cold_start:
        args.method = "euler_2"
        args.target = 1 / 255
        args.omega = 0.0

    if args.seed != 0:
        t.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.adv == "PGD":
        args.epochs = 100
        end = 35
        start = 5
        length  = 60
        eps_scheduler = SmoothedScheduler(args.target, "start={0},length={1},mid=0.6,beta=4.0".format(start,length))

    train_loader,test_loader,norm_mean,norm_std = load_dataset(args)
    
    dimss = 32
    if args.dataset in ["CIFAR10"]:
        dimss = 32
        odefunc = ODEfunc_abstract(dimss,dimss,act = "relu",nogr = args.nogr)
        ODE = ODEBlock(odefunc,args.method, 0,[0,args.endtime])
        model = ODENet_mini_CIFAR(ODE,device,norm_mean,norm_std)
    else:
        odefunc = ODEfunc_abstract(32,dimss,act = "relu",nogr = args.nogr)
        ODE = ODEBlock(odefunc,args.method, 0,[0,args.endtime])
        model = ODENet_mini_MNIST(ODE,device,norm_mean,norm_std)


    criterion = nn.CrossEntropyLoss()
    model.to(device)
    print("MODEL;params",sum(p.numel() for p in model.parameters() if p.requires_grad))

    if not args.cold_start and not args.no_warm_up:
        resume = args.exppath  + "{2}activation_{0}_method_{1}_checkpoint_epoch_{3}_seed_{4}.pth.tar".format(args.act,args.method,args.adv,args.epochs,args.seed)
        
        if args.adv == "box":
            resume = args.exppath + "boxactivation_mini_method_euler_2_epoch_50_seed_{0}_target_1_gamma_0.01_omega_0.0_traj_0_run_1.0.pth.tar".format(args.seed)


        if os.path.isfile(resume) and not args.adv in ["box_lin","lin","fixed",""] :
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = t.load(resume,map_location = device)
            args.start_epoch = checkpoint['epoch']
            if args.adv in ["box","box_trades"] and args.traj_mode == 0:
                args.start_epoch = 0
            best_acc1 = checkpoint['best_acc1'] 
            if "net.5.running_mean_step" in model.state_dict().keys() and "net.5.running_mean_step" not in checkpoint['state_dict'].keys():
                checkpoint['state_dict']["net.5.running_mean_step"] = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            best_acc1= 0 # start from scratch

    if args.adv in ["box"]:
        start = np.max(args.start_epoch,args.start)
        end = 40
        length = args.epochs - start - end
        length  = 60
        if args.dataset == "CIFAR10":
            box_model = Sequential.from_concrete_network(model.net,(3,32,32))
            cold_start_value = 0.1/255
        else:
            box_model = Sequential.from_concrete_network(model.net,(1,28,28))
            cold_start_value = 1/255

        increase_traj_list = [start,start + 25,start + 50]
        decrease_traj_list = [start + 65]

        if args.kappa > 0:
            box_model[5].allowed_trajectories = args.kappa
            increase_traj_list = []
            decrease_traj_list = []


        eps_scheduler = SmoothedScheduler(args.target, "start={0},length={1},mid=0.6,beta=4.0".format(start,length))

        optimizer = t.optim.Adam(box_model.parameters() ,args.lr,weight_decay=args.wd)


        if args.cold_start:
            assert args.method == "euler_2"
            increase_traj_list = []
            decrease_traj_list = []
            args.epochs = 50
            start = 10
            end = 10
            length = args.epochs - start - end
            eps_scheduler = SmoothedScheduler(cold_start_value, "start={0},length={1},mid=0.6,beta=4.0".format(start,length))
            optimizer = t.optim.Adam(box_model.parameters() ,args.lr,weight_decay=args.wd)
        else:
            box_model[5].update_qs(args.q_start,args.q_end)
            box_model[5].provable_training_mode = args.ptm   
    else:
        optimizer = t.optim.Adam(model.parameters(),args.lr,weight_decay=args.wd)



    scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[130,145] ,gamma=0.2)

    for epoch in range(args.start_epoch,args.epochs):
        if epoch == args.epochs -1 :
            args.final = True
        print("Training Epoch {0}:".format(epoch))

        tepoch_s = time.time()
        optimizer.zero_grad()

        if args.adv in [""]:
            loss, acc_train,stat_list =train(model,criterion,train_loader,optimizer,device,args)
        elif args.adv == "PGD":
            eps_scheduler.set_epoch_length(1)
            eps_scheduler.step_epoch()
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
            print("EPS:",eps)
            if epoch <start:
                loss, acc_train,stat_list =train(model,criterion,train_loader,optimizer,device,args)
            else:
                adversary = torchattacks.PGD(model,eps= eps,steps = 10,alpha = eps/5) #high c approx. small epsilon
                loss, acc_train,stat_list =train_adv(model,criterion,train_loader,optimizer,device,adversary,args)
        
        elif args.adv in ["box"]:
            eps_scheduler.set_epoch_length(1)
            eps_scheduler.step_epoch()
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
            if epoch < start:
                loss, acc_train,stat_list =train(box_model,criterion,train_loader,optimizer,device,args)
                box_mean = 0
                numb_t = 0
                numb_s = 0
            else:
                print(eps)
                if args.traj_mode == 0:
                    if (epoch) in increase_traj_list:
                        box_model[5].allowed_trajectories *=2
                    if epoch in decrease_traj_list:
                        box_model[5].allowed_trajectories = 4
                elif epoch  ==  60:
                    box_model[5].allowed_trajectories = args.traj_mode
                
                loss, acc_train,stat_list ,box_mean,numb_t,numb_s =train_box_provable(box_model,criterion,train_loader,optimizer,eps,device,args)  
        else:
            print("not implemented yet ",args.adv)
            exit()

        scheduler.step()

        tepoch_e = time.time()
        mins = (tepoch_e-tepoch_s)//60
        secs = (tepoch_e-tepoch_s)%60
        print('Train loss: {:.5f}'.format(loss), ", train accuracy:{:.4f} ".format(acc_train))
        if len(stat_list) == 3:
            labels = ['Total Steps:','Accepted Steps:','Average Step Size:']
        else:
            labels = ['Total Steps:','Normally Accepted Steps:','Increased Accepted Steps:','Rejected Steps:','Average Step Size:','Increased First Step:','Change Factor Inc:','Change Factor Dec:', 'Rejects after increase:','Error Estimate']

        loss_tensor = t.tensor(loss).cuda()                
        writer.add_scalar('trainingtime_epoch_min',mins,epoch)
        writer.add_scalar('training_loss',loss,epoch)
        writer.add_scalar('training_acc',acc_train,epoch)

        if args.adv in ["box"]:  
            writer.add_scalar('eps',eps,epoch)
            writer.add_scalar('splits',numb_s,epoch)
            writer.add_scalar('trajectories',numb_t,epoch)
            writer.add_scalar('boxes',box_mean,epoch)

        if args.act not in ["baseline","baseline_sin","baseline_sin_10_ps","baseline_sin_10_ns","baseline_10_ps","baseline_10_ns","mini_base","mini_adj"]:
            writer.add_scalar('training_error',stat_list[-1],epoch)
            writer.add_scalar('trainingsteps',stat_list[0],epoch)
            for i in range(np.minimum(len(stat_list),4)):
                print(labels[i],stat_list[i])


        test_acc,stat_list = test(model,test_loader,device,args,256)
        best_acc1 = test_acc
        writer.add_scalar('testing_acc',test_acc,epoch)
        print("Test Accuracy: {:.4f}".format(test_acc))

        if args.act not in ["baseline","baseline_sin","baseline_sin_10_ps","baseline_sin_10_ns","baseline_10_ps","baseline_10_ns","mini_base","mini_adj"]:
            writer.add_scalar('testing_steps',stat_list[0],epoch)
            writer.add_scalar('testing_error',stat_list[-1],epoch)
            for i in range(len(stat_list)):
                print(labels[i],stat_list[i])

        if (epoch+1) in [1,args.epochs]:
            if args.adv in ["box"]:
                if args.method == "euler_2":
                    args.target = 1
                save_checkpoint({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "{2}activation_{0}_method_{1}_epoch_{3}_seed_{4}_target_{5}_gamma_{6}_omega_{7}_traj_{8}_run_{9}.pth.tar".format(args.act,args.method,args.adv,(epoch+1),args.seed,args.target,args.gamma,args.omega,args.traj_mode,args.psi)) 
            else:
                save_checkpoint({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "{2}activation_{0}_method_{1}_checkpoint_epoch_{3}_seed_{4}.pth.tar".format(args.act,args.method,args.adv,(epoch+1),args.seed))


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0],*self.shape)

if __name__ == '__main__':
    main()



















