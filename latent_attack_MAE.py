import numpy as np
import os
import time
import argparse
parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--epochs', type=int, default=15)
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
parser.add_argument('--Nu', default=0.1, type=float)
parser.add_argument('--Delta', default=0.01, type=float)

parser.add_argument('--lamda', default=1.0, type=float)
parser.add_argument('--psi', default=1.0, type=float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--omega', default=3000.0, type=float)
parser.add_argument('--clip', default=1000.0, type=float)
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--debug', default = 0, type = int)
parser.add_argument('--dyn', default = 1, type = int)
parser.add_argument('--start', default = 0, type = int)
parser.add_argument('--target', default = 1.5, type = float)
parser.add_argument('--dataset', type = str, default = "physionet")
parser.add_argument('--traj_mode', default = 0, type = int)
parser.add_argument('--eps', default = 0.05, type = float)
parser.add_argument('--quantization', type=float, default=0.05, help="Quantization on the physionet dataset."
    "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
parser.add_argument('--cel_factor', type=float, default=0.)
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
parser.add_argument('-l', '--latents', type=int, default=20, help="Size of the latent state")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=2, help="Number of layers in ODE func in generative ODE")
parser.add_argument('-u', '--units', type=int, default=40, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=50, help="Number of units per layer in each of GRU update networks")
parser.add_argument('--rec-dims', type=int, default=40, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--ratio', type=float, default=1., help="up until which timepoint do we observe data w.r.t. 48 hours")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--load', action='store_true', help="Include  for hospiral mortality")
parser.add_argument('--dataratio', type=float, default=1)
parser.add_argument('--samples', default=400, type=int)
parser.add_argument('--steps', default=200, type=int)
parser.add_argument('--data_mode', type=int, default=1, help="Size of the latent state")
parser.add_argument('--run_backwards', action='store_true', help="run ts backwards")
parser.add_argument('--enc_factor', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1.0)
parser.add_argument('--pre_trained',action = 'store_true')
parser.add_argument('--only_std',action = 'store_true')
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
from PARC.AIDomains.abstract_layers import Sequential,Linear
from PARC.examples.eps_scheduler import SmoothedScheduler
from physionet.parse_datasets import parse_datasets
from utils import save_checkpoint,check_sound,test,update_statistics,train,train_adv, train_box_provable,load_dataset
from sklearn import metrics

def main():
    if args.gpu <= 2:
        device = t.device('cuda:0')
    else:
        device = t.device("cpu")

    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.pre_trained:
        args.exppath = args.exppath + "pre-trained/"

    args.exppath = args.exppath + args.dataset + "/"
    data_obj = parse_datasets(args, device)

    model = LatentODE(args,data_obj["input_dim"],data_obj["input_init"],device,1)
    z0_prior =torch.distributions.normal.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    if args.adv =="box":
        resume = args.exppath + "best_val_boxmet_{0}_act_{1}_bs_{2}_quantization_{3}_checkpoint_seed_{4}_omega_{5}_target_{6}_enc_factor_{7}_wd_{8}_data_mode_{9}.pth.tar".format(args.method,args.act,args.batch_size,args.quantization,args.seed,args.omega,args.target,args.enc_factor,args.wd,args.data_mode)
    else:
        args.wd = 1e-4
        resume = args.exppath + "best_val_{0}met_{1}_act_{2}_bs_{3}_quantization_{4}_checkpoint_seed_{5}_omega_{6}_target_{7}_enc_factor_{8}_wd_{9}_0.3_data_mode_{10}.pth.tar".format(args.adv,args.method,args.act,args.batch_size,args.quantization,args.seed,args.omega,args.target,args.enc_factor,args.wd,args.data_mode)
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
        exit()


    optimizer = t.optim.Adam(model.parameters(),args.lr,weight_decay=1e-4)

    scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000] ,gamma=0.2)
    test_loader =data_obj["test_dataloader"]
    model.eval()
    mae_list = []
    count = 0
    r11 = 0
    r12 = 0
    r15 = 0
    mae_list_adv = []
    attackers = []
    attackers.append(torchattacks.PGD(model, eps= 0.05,steps = args.steps,alpha = 0.1 / 40))

    attackers.append(torchattacks.PGD(model, eps= 0.1,steps = args.steps,alpha = 0.1 / 40)) 

    attackers.append(torchattacks.PGD(model, eps= 0.2,steps = args.steps,alpha = 0.1 / 40))

    adversarial_lists = [[0]*args.samples for _ in range(len(attackers) + 1)]

    epsi = args.eps
    for jjj in range(len(attackers)):
        if jjj > 0:
            print(np.round(np.mean(adversarial_lists[0]),3))
            if args.only_std:
                break
        adv = attackers[jjj]
        for batch_idx, data in tqdm(enumerate(test_loader),  total=len(test_loader)):
            if len(data['tp_to_predict']) < 1:
                continue
            if batch_idx == args.samples:
                break


            x = torch.cat((data["observed_data"],data["observed_mask"]),dim=2).float()

            pred,info = model.forward(x,data['observed_tp'].to(device),data['tp_to_predict'].to(device),data['observed_init'],target_mask = data['mask_predicted_data'],run_backwards = args.run_backwards)

            truth = data["data_to_predict"]

            mask = data["mask_predicted_data"]

            pred = pred.squeeze(2)
            idx_mask = torch.where(mask ==1)
            temp = (pred[idx_mask] - truth[idx_mask])
            mae = temp.abs().mean()

            adversarial_lists[0][batch_idx] = mae.item()

            if args.only_std:
                continue

            if args.run_backwards:
                adv_images, adv_init = adv(data, data["data_to_predict"],105)
            else:
                adv_images, adv_init = adv(data, data["data_to_predict"],103)

            x_adv = torch.cat((adv_images,data["observed_mask"]),dim=2).float()
            pred_adv,info = model.forward(x_adv,data['observed_tp'],data['tp_to_predict'],adv_init,target_mask = data['mask_predicted_data'],run_backwards = args.run_backwards)
            pred_adv = pred_adv.squeeze(2)
            temp_adv = (pred_adv[idx_mask] - truth[idx_mask])
            mae_adv = temp_adv.abs().mean()
            adversarial_lists[jjj+1][batch_idx] = mae_adv.item()


    for i in range(len(adversarial_lists)):
        if i == 0:
            print("STANDARD MAE:",np.round(np.mean(adversarial_lists[i]),3))
            if args.only_std:
                exit()
        else:
            print("ATTACKED MAE with perturbation eps = {0}:".format(attackers[i-1].eps),np.round(np.mean(adversarial_lists[i]),3))


    print("training, seed, data mode, target: ",args.adv,args.seed,args.data_mode,args.target)


    nu= 1 + args.Nu
    delta = args.Delta
    for jjj in range(1, len(adversarial_lists)):
        verified = 0
        for iii in range(len(adversarial_lists[0])):
            if adversarial_lists[jjj][iii] < nu * adversarial_lists[0][iii] + delta:
                verified += 1
        print("Adversarial robustness [%] nu={0},delta={1},eps{2}:".format(args.Nu,args.Delta,attackers[jjj-1].eps),verified/args.samples * 100)

    res1 = args.exppath + "adversarial_results/"
    if not os.path.isdir(res1):
        os.makedirs(res1)
    resume = args.exppath + "adversarial_results/{3}data_mode_{0}_seed_{1}_target_{2}.tar.pth".format(args.data_mode,args.seed,args.target,args.adv)
    torch.save({"list":adversarial_lists},resume)



def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):

    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:

        gaussian = torch.distributions.Independent(torch.distributions.normal.Normal(loc = mu_2d, scale = obsrv_std.repeat(n_data_points)), 1)

        log_prob = gaussian.log_prob(data_2d) 

        log_prob = log_prob / n_data_points 

    else:

        log_prob = torch.zeros([1]).to(data_2d.device).squeeze()

    return log_prob  


def compute_masked_likelihood(mu, data, mask, obsrv_std):

    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements

    n_traj, n_timepoints, n_dims = data.size()


    res = []


    for k in range(n_traj):

        for j in range(n_dims):

            data_masked = torch.masked_select(data[k,:,j], mask[k,:,j].bool())

            mu_masked = torch.masked_select(mu[k,:,j], mask[k,:,j].bool())

            log_prob = gaussian_log_likelihood(mu_masked, data_masked, obsrv_std)

            res.append(log_prob)


    res = torch.stack(res, 0).to(data.device)

    res = res.reshape((n_traj, n_dims))

    res = torch.mean(res, -1)

    return res


if __name__ == '__main__':

    main()
















