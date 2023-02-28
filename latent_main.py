import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
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
parser.add_argument('--omega', default=3000.0, type=float)
parser.add_argument('--delta', default=0.66, type=float)
parser.add_argument('--clip', default=1000.0, type=float)

parser.add_argument('--seed', default = 1, type = int)
parser.add_argument('--debug', default = 0, type = int)
parser.add_argument('--dyn', default = 1, type = int)
parser.add_argument('--start', default = 0, type = int)
parser.add_argument('--target', default = 1.5, type = float)
parser.add_argument('--dataset', type = str, default = "physionet")
parser.add_argument('--traj_mode', default = 0, type = int)
parser.add_argument('--kl_coef', default = 0.1, type = float)


parser.add_argument('--quantization', type=float, default=0.05, help="Quantization on the physionet dataset."
    "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
parser.add_argument('--cel_factor', type=float, default=0.)
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
parser.add_argument('-l', '--latents', type=int, default=20, help="Size of the latent state")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in ODE func in recognition ODE") #TODO
parser.add_argument('--gen-layers', type=int, default=2, help="Number of layers in ODE func in generative ODE")#TODO
parser.add_argument('-u', '--units', type=int, default=40, help="Number of units per layer in ODE func")#TODO
parser.add_argument('-g', '--gru-units', type=int, default=50, help="Number of units per layer in each of GRU update networks")
parser.add_argument('--rec-dims', type=int, default=40, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--ratio', type=float, default=1., help="up until which timepoint do we observe data w.r.t. 48 hours")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--load', action='store_true', help="Include  for hospiral mortality")
parser.add_argument('--dataratio', type=float, default=1)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--use_pred', action='store_true', help="Include extrapolated points")
parser.add_argument('--cold_start', action='store_true', help="Include extrapolated points")
parser.add_argument('--run_backwards', action='store_true', help="run ts backwards")
parser.add_argument('--full_data', action='store_true', help="run ts backwards")
parser.add_argument('--enc_factor', type=float, default=0.1)

parser.add_argument('--noise', type=float, default=0.3)

parser.add_argument('--data_mode', type=int, default=1, help="0: predict last tp with all data except last tp, 1: at least 6h forecast,2: at least 12h forecast,3: at least 24h forecast")


#FIRST set device before loading torch
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

import torch as t
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from latent_model import *

import pdb
import torchvision
import torchattacks
from torch.utils.tensorboard import SummaryWriter

from PARC.AIDomains.zonotope import HybridZonotope
from PARC.AIDomains.abstract_layers import Sequential,Linear, Encoder_ODE_RNN
from PARC.AIDomains.abstract_layers import DiffeqSolver as abstract_DiffeqSolver
from PARC.examples.eps_scheduler import SmoothedScheduler
from physionet.parse_datasets import parse_datasets
from utils import save_checkpoint,check_sound,test,update_statistics,train,train_adv, train_box_provable,load_dataset
from sklearn import metrics
from latent_utils import latent_train_epoch, latent_test,compute_masked_likelihood, gaussian_log_likelihood,latent_train_provable_epoch, latent_test_provable
def main():
    device = t.device('cuda:0') 
    args.exppath = args.exppath + args.dataset + "/"
    

    writer = SummaryWriter(args.exppath+ '/runs/{7}enc_factor_{0}_act_{1}_method_{2}_mae_{3}_bs_{4}_seed_{5}_target_{6}_data_mode_{8}_wd_{9}_dataset_{10}_psi_{11}'.format(args.enc_factor,args.act,args.method,args.omega,args.batch_size,args.seed,args.target,args.adv,args.data_mode,args.wd,args.dataset,args.psi)) 

    t.manual_seed(args.seed)
    np.random.seed(args.seed)


    data_obj = parse_datasets(args, device)
    train_loader = data_obj["train_dataloader"]
    test_loader =data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]


    model = LatentODE(args,data_obj["input_dim"],data_obj["input_init"],device,1)

    if args.act == "baseline_RNN":
        model = Baseline_RNN(args,data_obj["input_dim"],data_obj["input_init"],device,1)

    z0_prior =torch.distributions.normal.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    criterion = nn.BCEWithLogitsLoss()
    model.to(device)


    if args.load :
        resume = args.exppath + "bs_{0}_cel_factor_{1}_quantization_{2}_ratio_{5}_checkpoint_epoch_{3}_seed_{4}_dataratio_{6}.pth.tar".format(args.batch_size,args.cel_factor,args.quantization,20,args.seed,args.ratio,args.dataratio)
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = t.load(resume,map_location = device)
            args.start_epoch = checkpoint['epoch']
            if args.adv == "box" and args.traj_mode == 0:
                args.start_epoch = 0
            if "net.5.running_mean_step" in model.state_dict().keys() and "net.5.running_mean_step" not in checkpoint['state_dict'].keys():
                checkpoint['state_dict']["net.5.running_mean_step"] = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    print("MODEL;params",sum(p.numel() for p in model.parameters() if p.requires_grad))

    val_score = 1000
    val_epoch = 0
    val_start = 0
    flagge = False
    test_best_val_score = 1000
    reset_sched = []
    check_poins = [1500,1550,1600,1650,1700,1750]


    optimizer = t.optim.Adam(model.parameters(),args.lr,weight_decay=args.wd)
    scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000] ,gamma=0.2)


    for epoch in range(args.start_epoch,args.epochs):

        if args.adv in [""]:
            train_loss, train_acc, train_mae,F1,F2, Precision, Recall,Specifity, g_mean = latent_train_epoch(train_loader,criterion,args,model,device,epoch,optimizer,args.run_backwards)
            print('Train MAE :',train_mae)

            writer.add_scalar('training_loss',train_loss,epoch)
            writer.add_scalar('training_acc',train_acc,epoch)
            writer.add_scalar('training_MAE',train_mae,epoch)
            writer.add_scalar('training_g-mean',g_mean,epoch)

            if args.dataset == "physionet":

                test_loss,test_acc,test_mae, F1,F2, Precision, Recall,Specifity, g_mean = latent_test(test_loader,criterion,args,model,device,epoch,args.run_backwards)

                print('TEST MAE :',test_mae)
                print("MAE,Precision,Sensitivity/Recall,Specifity,g_mean, F1,F2,acc:{0:.3f},{1:.3f} ,{2:.3f} ,{3:.3f} ,{4:.3f} ,{5:.3f} ,{6:.3f},{7:.3f} ".format(test_mae,Precision,Recall,Specifity,g_mean,F1,F2,test_acc))
                writer.add_scalar('testing_loss',test_loss,epoch)
                writer.add_scalar('testing_MAE',test_mae,epoch)

                val_loss,test_acc,val_mae, F1,F2, Precision, Recall,Specifity, g_mean = latent_test(val_loader,criterion,args,model,device,epoch,args.run_backwards)
                print('VAL MAE :',val_mae)
                writer.add_scalar('val_loss',val_loss,epoch)
                writer.add_scalar('val_MAE',val_mae,epoch)

            if val_mae < val_score:
                val_score = val_mae
                val_epoch = epoch + 1
                test_best_val_score = test_mae

                save_checkpoint({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "best_val_{0}met_{1}_act_{2}_bs_{3}_quantization_{4}_checkpoint_seed_{5}_omega_{6}_target_{7}_enc_factor_{8}_wd_{9}_{10}_data_mode_{11}.pth.tar".format(args.adv,args.method,args.act,args.batch_size,args.quantization,args.seed,args.omega,args.target,args.enc_factor,args.wd,args.noise,args.data_mode))
            if val_epoch < epoch - 10 and not args.dataset == "toy":
                #model does not seem to improve anymore
                flagge = True
            
        scheduler.step()

        if (epoch + 1) in reset_sched:
            eps_scheduler = SmoothedScheduler(args.target, "start=0,length={0},mid=0.6,beta=4.0".format(length), init_value = 1e-6)

        if (epoch+1) in [args.epochs] or flagge or (epoch +1) in reset_sched or (epoch+1) in check_poins:
            if args.adv != "box":
                save_checkpoint({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "{0}met_{1}_act_{2}_bs_{3}_quantization_{4}_checkpoint_epoch_{5}_seed_{6}_omega_{7}_data_mode_{8}.pth.tar".format(args.adv,args.method,args.act,args.batch_size,args.quantization,(epoch+1),args.seed,args.omega,args.data_mode))
            else:
                save_checkpoint({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "{0}met_{1}_act_{2}_bs_{3}_quantization_{4}_checkpoint_epoch_{5}_seed_{6}_omega_{7}_target_{8}_enc_factor_{9}_wd_{10}_noise_{11}_w_bias_run_{12}_{13}.pth.tar".format(args.adv,args.method,args.act,args.batch_size,args.quantization,(epoch+1),args.seed,args.omega,args.target,args.enc_factor,args.wd,args.noise,args.psi,args.gamma))
                
            if flagge :
                print("early stopping")
                break

    print(args.ratio,args.cel_factor,args.seed,args.quantization,args.omega,args.adv)
    print(test_best_val_score,val_epoch,args.data_mode)
    if args.method == "dopri5_0.005_2a" and args.act != "baseline_RNN":
        print(model.diffeq_solver.running_mean_step)
        


if __name__ == '__main__':
    main()
















