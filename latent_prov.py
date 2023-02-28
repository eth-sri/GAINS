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
parser.add_argument('--adv', default="box", type=str,
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
parser.add_argument('--wd', type=float, default=1.0)
parser.add_argument('--use_pred', action='store_true', help="Include extrapolated points")
parser.add_argument('--cold_start', action='store_true', help="Include extrapolated points")
parser.add_argument('--enc_factor', type=float, default=0.1)
parser.add_argument('--run_backwards', action='store_true', help="reverse TS")

parser.add_argument('--data_mode', type=int, default=1, help="0: predict last tp with all data exept last tp, 1: at least 6h forecast,2: at least 12h forecast,3: at least 24h forecast")


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
from latent_utils import latent_train_epoch, latent_test, compute_masked_likelihood, gaussian_log_likelihood,latent_train_provable_epoch, latent_test_provable
def main():

    device = t.device('cuda:0') 

    writer = SummaryWriter(args.exppath+ '/runs/{7}enc_factor_{0}_act_{1}_method_{2}_mae_{3}_quant_{4}_seed_{5}_target_{6}_data_mode_{8}_wd_{9}'.format(args.enc_factor,args.act,args.method,args.omega,args.quantization,args.seed,args.target,args.adv,args.data_mode,args.wd)) 
    args.exppath = args.exppath + args.dataset + "/"
    
    if not os.path.isdir(args.exppath):
        os.makedirs(res1)
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

    print("MODEL;params",sum(p.numel() for p in model.parameters() if p.requires_grad))
    val_score = 1000
    val_epoch = 0

    if args.adv == "box" and args.act != "baseline_RNN":
        box_model = []
        box_model.append(Encoder_ODE_RNN.from_concrete_layer(model.encoder_z0,(data_obj["input_init"])))
        box_model.append(abstract_DiffeqSolver.from_concrete_layer(model.diffeq_solver,(1,model.z0_dim)))
        box_model.append(Sequential.from_concrete_network(model.decoder,(1,1,model.z0_dim)))

        args.epochs = 120
        abstract_ratio = 0.1
        start = 5
        length = 60
        val_start = 65
        ratios = [ 20,  25,  30,  35,  40, 45, 50, 55]
        #          0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1
        ratioss = [10,15]
        #         0.15,0,2

        eps_scheduler = SmoothedScheduler(args.target, "start={0},length={1},mid=0.6,beta=4.0".format(start,length))
        params = list(box_model[0].parameters()) + list(box_model[1].parameters()) + list(box_model[2].parameters())
        optimizer = t.optim.Adam(params,args.lr,weight_decay=args.wd)

    else:
        optimizer = t.optim.Adam(model.parameters(),args.lr,weight_decay=args.wd)

    scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000] ,gamma=0.2)


    for epoch in range(args.start_epoch,args.epochs):
        if args.adv in ["box"]:
            eps_scheduler.set_epoch_length(1)
            eps_scheduler.step_epoch()
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
            print("EPS",eps)
            if epoch in ratios:
                abstract_ratio += 0.1
            if epoch in ratioss:
                abstract_ratio += 0.05

            train_loss, train_box, train_mae, abstract_mae = latent_train_provable_epoch(train_loader,criterion,args,box_model,device,epoch,optimizer,eps,abstract_ratio,args.run_backwards)

            writer.add_scalar('eps',eps,epoch)
            writer.add_scalar('training_loss',train_loss,epoch)
            writer.add_scalar('training_MAE',train_mae,epoch)
            writer.add_scalar('training_box',train_box,epoch)
            writer.add_scalar('abstract_MAE',abstract_mae,epoch)

            test_mae = latent_test_provable(test_loader,box_model,device,args.run_backwards)
            writer.add_scalar('testing_MAE',test_mae,epoch)

        scheduler.step()

        if epoch >= val_start:
            val_mae = latent_test_provable(val_loader,box_model,device,args.run_backwards)
            if val_mae < val_score:
                val_score = val_mae
                val_epoch = epoch + 1
                save_checkpoint({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "best_val_{0}met_{1}_act_{2}_bs_{3}_quantization_{4}_checkpoint_seed_{5}_omega_{6}_target_{7}_enc_factor_{8}_wd_{9}_data_mode_{10}.pth.tar".format(args.adv,args.method,args.act,args.batch_size,args.quantization,args.seed,args.omega,args.target,args.enc_factor,args.wd,args.data_mode))

        if (epoch+1) in [args.epochs]:
            if args.adv == "rr":
                args.ratio = 1.0
            if args.adv != "box":
                save_checkpoint({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "{0}met_{1}_act_{2}_bs_{3}_quantization_{4}_checkpoint_epoch_{5}_seed_{6}_omega_{7}_data_mode_{8}.pth.tar".format(args.adv,args.method,args.act,args.batch_size,args.quantization,(epoch+1),args.seed,args.omega,args.data_mode))
            else:
                save_checkpoint({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},savepath = args.exppath, filename = "{0}met_{1}_act_{2}_bs_{3}_quantization_{4}_checkpoint_epoch_{5}_seed_{6}_omega_{7}_target_{8}_enc_factor_{9}_wd_{10}_data_mode_{10}.pth.tar".format(args.adv,args.method,args.act,args.batch_size,args.quantization,(epoch+1),args.seed,args.omega,args.target,args.enc_factor,args.wd))


    print(args.ratio,args.cel_factor,args.seed,args.quantization,args.omega,args.adv)
    if args.method == "dopri5_0.005_2a" and args.act != "baseline_RNN":
        print(model.diffeq_solver.running_mean_step)
        print(val_epoch)
        


if __name__ == '__main__':
    main()
















