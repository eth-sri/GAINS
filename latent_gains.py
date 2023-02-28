import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--eps',  default=0.00001, type=float)
parser.add_argument('--Nu',  default=0.1, type=float)
parser.add_argument('--Delta',  default=0.01, type=float)
parser.add_argument('--psi',  default=1.0, type=float)
parser.add_argument('--act', type = str, default = "mini")
parser.add_argument('--exppath', type = str, default = "models/")
parser.add_argument('--method', default="dopri5_0.005_2a", type=str)
parser.add_argument('--endtime', default=1.0, type=float)
parser.add_argument('--samples', default=400, type=int)
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--dataset', type = str, default = "physionet")
parser.add_argument('--seed', default = 12345, type = int)
parser.add_argument('--adv', default="", type=str,
                    help='indicates whether "" normal training or "CW" adv training')

parser.add_argument('--traj_mode', default = 0, type = int)
parser.add_argument('--target', default = 0.1, type = float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--omega', default=3000.0, type=float)
parser.add_argument('--zono', action='store_true')
parser.add_argument('--enc_steps', default = 200, type = int)


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
parser.add_argument('--ratio', type=float, default=1, help="up until which timepoint do we observe data w.r.t. 48 hours")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--load', action='store_true', help="Include  for hospiral mortality")
parser.add_argument('--dataratio', type=float, default=1)
parser.add_argument('--use_pred', action='store_true', help="Include extrapolated points")
parser.add_argument('--run_backwards', action='store_true', help="run ts backwards")
parser.add_argument('--enc_factor', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1.0)

parser.add_argument('--data_mode', type=int, default=1, help="Size of the latent state")
parser.add_argument('--pre_trained',action = 'store_true')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

import pandas as pd
import torch 
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from latent_model import *
import pdb
import torchvision
import torchattacks

from PARC.AIDomains.deeppoly import DeepPoly, backprop_dp, backward_deeppoly, compute_dp_relu_bounds, backprop_dp_with_bounds
from PARC.AIDomains.abstract_layers import Conv2d, Linear, ReLU, GlobalAvgPool2d, Flatten, BatchNorm2d, Upsample, Log, Exp, \
        Inv, LogSumExp, Entropy, BatchNorm1d, AvgPool2d, Bias, Scale, Normalization, BasicBlock, WideBlock, FixupBasicBlock, Sequential, TestBlock, GroupNorm, ConcatConv, ODEBlock_A, Encoder_ODE_RNN,Sequential_with_bounds
from PARC.AIDomains.abstract_layers import DiffeqSolver as abstract_DiffeqSolver
from PARC.AIDomains.zonotope import HybridZonotope
from physionet.parse_datasets import parse_datasets
from utils import save_checkpoint,check_sound,test,update_statistics,train,train_adv, train_box_provable,load_dataset
from sklearn import metrics
from latent_utils import latent_test_provable, latent_test

def main():
    device = torch.device('cuda:0') 

    #device = torch.device('cpu')
    args.extrap = True
    args.run_backwards = True

    if args.pre_trained:
        args.exppath = args.exppath + "pre-trained/"

    args.exppath = args.exppath + args.dataset + "/"

    data_obj = parse_datasets(args, device)
    model = LatentODE(args,data_obj["input_dim"],data_obj["input_init"],device,1)


    resume = args.exppath + "best_val_boxmet_{0}_act_{1}_bs_{2}_quantization_{3}_checkpoint_seed_{4}_omega_{5}_target_{6}_enc_factor_{7}_wd_{8}_data_mode_{9}.pth.tar".format(args.method,args.act,args.batch_size,args.quantization,args.seed,args.omega,args.target,args.enc_factor,args.wd,args.data_mode)

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume,map_location = device)
        args.start_epoch = checkpoint['epoch']
        if "diffeq_solver.running_mean_step" in model.state_dict().keys() and "diffeq_solver.running_mean_step" not in checkpoint['state_dict'].keys():
            checkpoint['state_dict']["diffeq_solver.running_mean_step"] = nn.parameter.Parameter(torch.ones(1)*0.25,requires_grad= False)

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        exit()

    test_loader =data_obj["test_dataloader"]

    model = model.to(device)
    print(model.diffeq_solver.running_mean_step)

    model.eval()
    model.double()

    verified = 0
    boxes_mean = 0
    acc = 0
    eps = args.eps


    #prepare abstract model
    encoder = model.encoder_z0
    classif = model.classifier

    abstract_enc = Encoder_ODE_RNN.from_concrete_layer(encoder,(data_obj["input_init"]))
    abstract_enc.mode = "abstract"
    abstract_enc.z0_diffeq_solver.mode = "abstract"

    abstract_diffeq = abstract_DiffeqSolver.from_concrete_layer(model.diffeq_solver,(1,model.z0_dim))
    abstract_diffeq.mode = "abstract"
    abstract_decoder = Sequential_with_bounds.from_concrete_layer(model.decoder,(1,1,model.z0_dim))

    box_model = [abstract_enc,abstract_diffeq,abstract_decoder]


    sound = 0 
    interval_len = []
    mae_list = []
    abstract_mae_list = []
    start = time.time()
    ratios = []

    [x.eval().double() for x in box_model]
    unsound = 0
    r15 = 0
    r12 = 0
    r11 = 0
    box_lengths = []

    running_mean_step = box_model[1].ode_func_train.running_mean_step.data.clone()
    c_l,d_l,a_l = [],[],[]

    epsilons = [0.05,0.1,0.2]
    verified_lists = [[0]*args.samples for _ in range(len(epsilons) + 1)]

    with torch.no_grad():
        for jjj in range(len(epsilons)):
            eps =  epsilons[jjj]
            for batch_idx, data in tqdm(enumerate(test_loader),  total=args.samples):
                if batch_idx == args.samples :
                    break

                x = torch.cat((data["observed_data"],data["observed_mask"]),dim=2).double().to(device)
                
                pred_concrete,info = model.forward(x,data['observed_tp'].to(device),data['tp_to_predict'].double().to(device),data['observed_init'].double(),target_mask = data['mask_predicted_data'].double(),run_backwards = args.run_backwards)
                pred_concrete = pred_concrete.squeeze(2)
                                
                obs_data_box, init_box = perturb_TS(data,eps,device)

                pred_abstract,info_abstract = forward_box_model(box_model,obs_data_box,data['observed_tp'].to(device),init_box,data['tp_to_predict'].double().to(device),data['mask_predicted_data'].double().to(device),None,run_backwards =args.run_backwards)

                pred_abstract = pred_abstract.squeeze(2)
                lb,ub = pred_abstract.concretize()

                #think about a way to refine
                idx = torch.where(data['mask_predicted_data'] > 0)
                lb,ub = lb[idx],ub[idx]

                box_int = (ub -lb).mean()
                box_lengths.append(box_int.item())

                #INIT DP
                bounds = box_model[2].bounds
                xx = 1
                for y in box_model[2].output_dim:
                    xx *= y
                #DP Decoder Output function
                expr_coef = torch.ones(xx).to(device)
                expr_coef = torch.diag(expr_coef).unsqueeze(0).type(torch.DoubleTensor).to(device)
                
                #dp = backprop_dp_with_bounds(box_model[2].net,  DeepPoly(expr_coef = expr_coef), it= 10, use_lambda=False, bounds = bounds)
                dp = DeepPoly(expr_coef = expr_coef)
                bounds = box_model[2].bounds

                #only lin layer, no need to do more here
                dp = dp_out_fct(box_model[2],dp,bounds,device)
                
                #DP decoder
                #permute needs to be adressed if more than 1 BS
                box_model[1].ode_func_train.running_mean_step.data = torch.clamp(running_mean_step,0.0,data['tp_to_predict'][-1])
                dp_0 = dp.clone()
                dp_1 = dp.clone()

                dp = backprop_dp(box_model[1].ode_func_train, dp, it = 10, use_lambda=False)

                dp_0 = backprop_dp(box_model[1].ode_func_train, dp_0, it = 17, use_lambda=False)

                dp_1 = backprop_dp(box_model[1].ode_func_train, dp_1, it = 18, use_lambda=False)

                box_model[1].ode_func_train.running_mean_step.data = running_mean_step

                #first make dp back through transform of encoder
                #take care of the fact that we also predict std
                dp = DeepPoly(torch.cat((dp.x_l_coef,torch.zeros_like(dp.x_l_coef)),dim = -1),torch.cat((dp.x_u_coef,torch.zeros_like(dp.x_u_coef)),dim = -1),dp.x_l_bias,dp.x_u_bias)
                dp_0 = DeepPoly(torch.cat((dp_0.x_l_coef,torch.zeros_like(dp_0.x_l_coef)),dim = -1),torch.cat((dp_0.x_u_coef,torch.zeros_like(dp_0.x_u_coef)),dim = -1),dp_0.x_l_bias,dp_0.x_u_bias)
                dp_1 = DeepPoly(torch.cat((dp_1.x_l_coef,torch.zeros_like(dp_1.x_l_coef)),dim = -1),torch.cat((dp_1.x_u_coef,torch.zeros_like(dp_1.x_u_coef)),dim = -1),dp_1.x_l_bias,dp_1.x_u_bias)
                
                dp = dp_encoder(box_model[0],dp,data["observed_tp"],data["observed_mask"],obs_data_box,device,it = 10,run_backwards=args.run_backwards)
                
                dp_0 = dp_encoder(box_model[0],dp_0,data["observed_tp"],data["observed_mask"],obs_data_box,device,it = 17,run_backwards=args.run_backwards)
                dp_1 = dp_encoder(box_model[0],dp_1,data["observed_tp"],data["observed_mask"],obs_data_box,device,it = 18,run_backwards=args.run_backwards)

                low,upp = dp.dp_concretize(bounds = init_box.concretize())
                low, upp = low.unsqueeze(0)[idx],upp.unsqueeze(0)[idx]

                low0,upp0 = dp_0.dp_concretize(bounds = init_box.concretize())
                low0, upp0 = low0.unsqueeze(0)[idx],upp0.unsqueeze(0)[idx]

                low1, upp1 = dp_1.dp_concretize(bounds = init_box.concretize())
                low1, upp1 = low1.unsqueeze(0)[idx],upp1.unsqueeze(0)[idx]

                low = torch.maximum(low,torch.maximum(low0,torch.maximum(low1,lb)))
                upp = torch.minimum(upp,torch.minimum(upp0,torch.minimum(upp1,ub)))



                #print((upp -low + 1e-10).mean().item())
                interval_len.append((upp-low).mean().item())


                truth = data["data_to_predict"][idx]
                mae = (pred_concrete[idx] - truth).abs().mean()

                abstact_low = (low- truth).abs()
                abstract_upp = (upp- truth).abs()

                abstract_mae = torch.maximum(abstact_low ,abstract_upp).mean()

                verified_lists[0][batch_idx] = mae.item()
                verified_lists[jjj+1][batch_idx] = abstract_mae.item()

                
            
    print(args.data_mode,args.target,args.seed)
    end = time.time()
    secs = round((end - start )/args.samples,2)
    print("AVERAGE TIME:            ",secs)

    for i in range(len(verified_lists)):
        if i == 0:
            print("STANDARD MAE:",np.round(np.mean(verified_lists[i]),3))
        else:
            print("Provable MAE:",np.round(np.mean(verified_lists[i]),3))

    nu= 1 + args.Nu
    delta = args.Delta

    for jjj in range(1, len(verified_lists)):
        verified = 0
        for iii in range(len(verified_lists[0])):
            if verified_lists[jjj][iii] < nu * verified_lists[0][iii] + delta:
                verified += 1
        print("Certified,nu=1.1,delta=0.01:",verified/args.samples)

    res1 = args.exppath + "gains_results/"
    if not os.path.isdir(res1):
        os.makedirs(res1)

    resume = args.exppath + "gains_results/{3}data_mode_{0}_seed_{1}_target_{2}.tar.pth".format(args.data_mode,args.seed,args.target,args.adv)
    torch.save({"list":verified_lists},resume)

def forward_box_model(model,x,obs_tp,init_data,tp_to_predict,target_mask,sampler,mode = 0,box = None,run_backwards = False):
    if box == None:
        first_point_mu, first_point_std = model[0](x,obs_tp, init_data,run_backwards = run_backwards)

        if (not model[0].training) or isinstance(x,HybridZonotope):
            first_point_enc = first_point_mu #maybe add some std to it
        else:
            first_point_enc = sampler.sample(first_point_mu.size()).squeeze(-1) * first_point_std + first_point_mu

        all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc) }
        if mode == 1:
            return first_point_enc, all_extra_info
    else:
        first_point_enc = box
        all_extra_info = {}

    sol_y = model[1](first_point_enc, tp_to_predict, mask=target_mask)

    if isinstance(sol_y,HybridZonotope):
        pred_x,model[2].bounds = model[2](sol_y,True)
    else:
        pred_x = model[2](sol_y)

    return pred_x, all_extra_info

def perturb_TS(data,eps,device):
    obs_data_low, obs_data_upp = data["observed_data"].clone() - eps, data["observed_data"].clone() + eps
    obs_data_low[torch.where(data["observed_mask"]==0)] *= 0
    obs_data_upp[torch.where(data["observed_mask"]==0)] *= 0
    #clamp 10 and 27
    obs_data_low[:,:,10], obs_data_upp[:,:,10] = torch.clamp(obs_data_low[:,:,10],0.0,1.0),torch.clamp(obs_data_upp[:,:,10],0.0,1.0)
    obs_data_low[:,:,28], obs_data_upp[:,:,28] = torch.clamp(obs_data_low[:,:,28],0.0,1.0),torch.clamp(obs_data_upp[:,:,28],0.0,1.0) 

    obs_data_low,obs_data_upp = torch.cat((obs_data_low,data["observed_mask"]),dim=2).double().to(device),torch.cat((obs_data_upp,data["observed_mask"]),dim=2).double().to(device)
            
    init_low, init_upp = data["observed_init"].clone().double(), data["observed_init"].clone().double()
    init_low[:,0], init_upp[:,0] = init_low[:,0] - eps , init_upp[:,0] + eps
    init_low[:,2], init_upp[:,2] = init_low[:,2] - eps , init_upp[:,2] + eps

    obs_data_box = HybridZonotope.construct_from_bounds(min_x = obs_data_low, max_x=obs_data_upp, domain="box",dtype = obs_data_upp.dtype)
    init_box = HybridZonotope.construct_from_bounds(min_x = init_low, max_x=init_upp, domain="box",dtype = init_upp.dtype)
    return obs_data_box,init_box

def dp_out_fct(out_fct,dp,bounds,device):
    return  backprop_dp_with_bounds(out_fct.net,  dp, it= 10, use_lambda=False, bounds = bounds)

def dp_encoder(encoder,dp,obs_tp,obs_mask,obs_data_box,device,it,run_backwards = False):
    if len(encoder.bounds) == 3:
        bounds = encoder.bounds[-1]
        dp = backprop_dp_with_bounds(encoder.transform_z0.net,  dp, it= it, use_lambda=False, bounds = bounds)

    #go back through ODE encoder part    
    bounds = encoder.bounds[1].copy()

    for jj in range(len(obs_tp)-1,-1,-1):

        if run_backwards:
            #print(len(obs_tp)-1 - jj)
            temp_bounds = bounds[len(obs_tp)-1 - jj].copy() 
            temp_data = obs_data_box[:,len(obs_tp)-1 - jj,0:obs_data_box.shape[-1]//2]
            mask = obs_mask[:,len(obs_tp)-1 - jj,:]
        else:
            temp_bounds = bounds[jj].copy() 
            temp_data = obs_data_box[:,jj,0:obs_data_box.shape[-1]//2]
            mask = obs_mask[:,jj,:]

        dp = dp.dp_GRU(temp_data,encoder.GRU_update.update_gate.net,encoder.GRU_update.reset_gate.net,encoder.GRU_update.new_state_net.net, mask, temp_bounds.pop(-1),it)
        ode_bounds, time_points = temp_bounds.pop(-1)
        if time_points == None:
            continue

        #seperate y mean and std part 
        mid = dp.x_l_coef.shape[-1]//2
        dp_y,dp_std = DeepPoly(dp.x_l_coef[:,:,0:mid],dp.x_u_coef[:,:,0:mid]), DeepPoly(dp.x_l_coef[:,:,mid:],dp.x_u_coef[:,:,mid:],dp.x_l_bias,dp.x_u_bias)

        for ii in range(len(time_points)-2,-1,-1):
            t0 = time_points[ii]
            dt = time_points[ii + 1] - time_points[ii]
            #dp_y = dp_y.dp_euler_step_with_bounds(encoder.z0_diffeq_solver.ode_func_f,t0,dt,ode_bounds.pop(-1))
            dp_y = dp_y.dp_euler_step_with_bounds(encoder.z0_diffeq_solver.ode_func_f,t0,dt,ode_bounds[ii],it)
        #combine again
        dp = DeepPoly(torch.cat((dp_y.x_l_coef,dp_std.x_l_coef),dim = -1),torch.cat((dp_y.x_u_coef,dp_std.x_u_coef),dim = -1),dp_y.x_l_bias + dp_std.x_l_bias,dp_y.x_u_bias + dp_std.x_u_bias)


    #go back initial layer
    if encoder.hidden_init != None:
        dp = backprop_dp_with_bounds(encoder.hidden_init.net,  dp, it= it, use_lambda=False, bounds = encoder.bounds[0])
    return dp 


if __name__ == '__main__':
    main()



























