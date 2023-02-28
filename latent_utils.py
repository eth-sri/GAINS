import numpy as np
import os
import time
import argparse
import torch
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

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def forward_box_model(model,x,obs_tp,init_data,tp_to_predict,target_mask,sampler, run_backwards = False):

    first_point_mu, first_point_std = model[0](x,obs_tp, init_data,run_backwards = run_backwards)
    if (not model[0].training) or isinstance(x,HybridZonotope):
        first_point_enc = first_point_mu 
    else:
        first_point_enc = sampler.sample(first_point_mu.size()).squeeze(-1) * first_point_std + first_point_mu
    sol_y = model[1](first_point_enc, tp_to_predict, mask=target_mask)
    pred_x = model[2](sol_y)
    all_extra_info = {"first_point": (first_point_mu, first_point_std, first_point_enc) }
    return pred_x, all_extra_info

def latent_train_provable_epoch(train_loader,criterion,args,model,device,epoch,optimizer,eps,abstract_ratio,run_backwards):
    print("Training Epoch {0}:".format(epoch))
    tepoch_s = time.time()
    optimizer.zero_grad()
    mae_list = []
    abstract_mae_list = []
    acc = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    lamda = eps/args.target
    box_losses = []
    time_tps = []

    loss_list = []
    prior = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))

    [x.train() for x in model]
    for batch_idx, data in tqdm(enumerate(train_loader),  total=len(train_loader)):
        start_t = time.time() 
        truth = data["data_to_predict"]
        mask = data["mask_predicted_data"]
        time_tps.append(len(data['observed_tp']))
        if data['observed_init'] != None:
            data['observed_init'] = data['observed_init'].to(device)

        x_concrete= torch.cat((data["observed_data"],data["observed_mask"]),dim=2).float().to(device)
        pred_concrete,info = forward_box_model(model,x_concrete,data['observed_tp'].to(device),data['observed_init'],data['tp_to_predict'].to(device),data['mask_predicted_data'].to(device),prior,run_backwards = run_backwards)
        #KL_loss
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_distr = torch.distributions.normal.Normal(fp_mu, fp_std.abs() + 1e-7)
        kldiv_z0 = torch.distributions.kl_divergence(fp_distr, prior).mean(2).squeeze()
        kldiv_z0 = kldiv_z0.mean()

        pred_concrete = pred_concrete.squeeze(2)

        res = compute_masked_likelihood(pred_concrete, truth, mask, torch.tensor([0.01]).to(device))
        res = res.mean()

        wait_until_kl_inc = 10
        itr = (epoch * len(train_loader) + batch_idx) 
        if itr // len(train_loader) < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (itr // len(train_loader) - wait_until_kl_inc))
        enc_loss = - (res -  kl_coef* kldiv_z0) * args.enc_factor

        idx_mask = torch.where(mask ==1)
        temp = (pred_concrete[idx_mask] - truth[idx_mask])
        mae = temp.abs().mean()
        mae_list.append(mae.item())


        concrete_loss = enc_loss + args.omega * mae
        #abstract setting
        if eps > 0:
            #here we don't sample/consider std
            obs_data_box, init_box = perturb_TS(data,eps,device,args,abstract_ratio,run_backwards)

            pred_abstract,info_abstract = forward_box_model(model,obs_data_box,data['observed_tp'].to(device),
                        init_box,data['tp_to_predict'].to(device),data['mask_predicted_data'].to(device),prior,run_backwards = run_backwards)

            pred_abstract = pred_abstract.squeeze(2)

            likelihood_mu = pred_abstract.get_gaussian_likelihood_dist(truth)
            
            abstract_res = compute_masked_likelihood(likelihood_mu, truth, mask, torch.tensor([0.01]).to(device))
            abstract_res = abstract_res.mean()

            abstract_fp_mu, abstract_fp_std, abstrac_fp_enc = info_abstract["first_point"]
            low, upp = abstract_fp_mu.concretize()
            kl_mu = torch.where(upp.abs() >= low.abs(), upp,low)
            kl_std = abstract_fp_std.get_gaussian_kl_std()
            fp_distr = torch.distributions.normal.Normal(kl_mu, kl_std)
            abstract_kldiv_z0 = torch.distributions.kl_divergence(fp_distr, prior).mean(2).squeeze()
            abstract_kldiv_z0 = abstract_kldiv_z0.mean()

            abstract_enc_loss = - (abstract_res -  kl_coef* abstract_kldiv_z0) * args.enc_factor

            abstract_mae = (likelihood_mu[idx_mask] - truth[idx_mask]).abs().mean() 
            abstract_mae_list.append(abstract_mae.item())

            abstract_loss = abstract_mae * args.omega + abstract_enc_loss

            low,upp = pred_abstract.concretize()
            box_loss = (upp[idx_mask] - low[idx_mask]).mean()
            box_losses.append(box_loss.item())
            

            loss = (1-lamda * args.delta) * concrete_loss + lamda * args.delta *abstract_loss  + args.gamma * box_loss
        else:
            
            loss = (1-lamda * args.delta)*concrete_loss
            box_losses.append(0)
            abstract_mae_list.append(mae.item())


        loss_list.append(loss.item()) 
        loss.backward()
        params = list(model[0].parameters()) + list(model[1].parameters()) + list(model[2].parameters())
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        optimizer.zero_grad()



    tepoch_e = time.time()
    mins = (tepoch_e-tepoch_s)//60
    secs = (tepoch_e-tepoch_s)%60
    print("average num. of obs tp", np.mean(time_tps))
    if eps > 0:
        print("BOX LOSS:", np.mean(box_losses))
    print('Train loss: {:.5f}'.format(np.mean(loss_list)))
    print('Train MAE :',np.mean(mae_list))

    return np.mean(loss_list),np.mean(box_losses), np.mean(mae_list), np.mean(abstract_mae_list)

def latent_test_provable(test_loader,model,device,run_backwards = False):
    mae_list = []
    [x.eval() for x in model]
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader),  total=len(test_loader)):
            truth = data["data_to_predict"]
            mask = data["mask_predicted_data"]
            if data['observed_init'] != None:
                data['observed_init'] = data['observed_init'].to(device)

            x_concrete= torch.cat((data["observed_data"],data["observed_mask"]),dim=2).float().to(device)
            pred_concrete,info = forward_box_model(model,x_concrete,data['observed_tp'].to(device),data['observed_init'],data['tp_to_predict'].float().to(device),data['mask_predicted_data'].to(device),sampler=None,run_backwards = run_backwards)

            pred_concrete = pred_concrete.squeeze(2)
            idx_mask = torch.where(mask ==1)
            temp = (pred_concrete[idx_mask] - truth[idx_mask])
            mae = temp.abs().mean()
            mae_list.append(mae.item())

    print('TEST MAE :',np.mean(mae_list))
    return np.mean(mae_list)



def latent_train_epoch(train_loader,criterion,args,model,device,epoch,optimizer,run_backwards):
    print("Training Epoch {0}:".format(epoch))
    tepoch_s = time.time()
    optimizer.zero_grad()
    mae_list = []
    acc = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    loss_list = []
    model.train()
    for batch_idx, data in tqdm(enumerate(train_loader),  total=len(train_loader)):
        if not args.dataset == "toy":
            init = data['observed_init'].float()
        else:
            init = None
        x = torch.cat((data["observed_data"],data["observed_mask"]),dim=2).float().to(device)
        pred,info = model.forward(x,data['observed_tp'].to(device),data['tp_to_predict'].float().to(device),init,target_mask = data['mask_predicted_data'].float(),run_backwards = run_backwards)
        #KL_loss
        fp_mu, fp_std, fp_enc = info["first_point"]

        fp_distr = torch.distributions.normal.Normal(fp_mu, fp_std.abs() + 1e-7)
        kldiv_z0 = torch.distributions.kl_divergence(fp_distr, model.z0_prior).mean(2).squeeze()
        #check
        kldiv_z0 = kldiv_z0.mean()

        truth = data["data_to_predict"]
        mask = data["mask_predicted_data"]
        pred = pred.squeeze(2)
        
        res = compute_masked_likelihood(pred, truth, mask, torch.tensor([0.01]).to(device))
        res = res.mean()


        wait_until_kl_inc = 10
        itr = (epoch * len(train_loader) + batch_idx) 
        if itr // len(train_loader) < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (itr // len(train_loader) - wait_until_kl_inc))
        enc_loss = - (res -  kl_coef* kldiv_z0) * args.enc_factor
        
        #masked MSE
        idx_mask = torch.where(mask ==1)
        temp = (pred[idx_mask] - truth[idx_mask])
        mae = temp.abs().mean()
        mae_list.append(mae.item())

        loss = enc_loss + args.omega * mae

        loss_list.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()
        optimizer.zero_grad()

    tepoch_e = time.time()
    mins = (tepoch_e-tepoch_s)//60
    secs = (tepoch_e-tepoch_s)%60

    print('Train loss: {:.5f}'.format(np.mean(loss_list)))


    return np.mean(loss_list),acc /3200,np.mean(mae_list),0,0, 0, 0,0,0

def latent_test(test_loader,criterion,args,model,device,epoch = 15,run_backwards = False):
    mae_list = []
    acc = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    if args.adv == "rr":
        args.ratio = 1.0
    loss_list = []

    time_tps = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader),  total=len(test_loader)):
            x = torch.cat((data["observed_data"],data["observed_mask"]),dim=2).float().to(device)

            time_tps.append(len(data['tp_to_predict']))
            pred,info = model.forward(x,data['observed_tp'].to(device),data['tp_to_predict'].float().to(device),data['observed_init'],target_mask = data['mask_predicted_data'],run_backwards = run_backwards)
            truth = data["data_to_predict"]

            mask = data["mask_predicted_data"]

            pred = pred.squeeze(2)
            idx_mask = torch.where(mask ==1)

            temp = (pred[idx_mask] - truth[idx_mask])
            mae = temp.abs().mean()
            mae_list.append(mae.item())

            fp_mu, fp_std, fp_enc = info["first_point"]
            fp_distr = torch.distributions.normal.Normal(fp_mu, fp_std.abs() + 1e-7)
            kldiv_z0 = torch.distributions.kl_divergence(fp_distr, model.z0_prior).mean(2).squeeze()
            kldiv_z0 = kldiv_z0.mean()

            truth = data["data_to_predict"]
            res = compute_masked_likelihood(pred, truth, mask, torch.tensor([0.01]).to(device))
            res = res.mean()

            kl_coef = 1.
            enc_loss = - (res -  kl_coef* kldiv_z0) * args.enc_factor
            
            if args.classif:
                labels = data["labels"].squeeze()
                pred_labels = info["label_predictions"].squeeze()
                cel_loss = criterion(pred_labels,labels)

                preds = torch.where(pred_labels > 0,1.,0.)
                acc += torch.sum(preds == labels).item()
                TP += ((preds == 1) * (labels == 1)).sum()
                FN += ((preds == 0) * (labels == 1)).sum()
                FP += ((preds == 1) * (labels == 0)).sum()
                TN += ((preds == 0) * (labels == 0)).sum()
                loss = args.cel_factor * cel_loss + enc_loss
            else:
                loss = enc_loss
            loss +=  args.omega * mae
            #pdb.set_trace()
            loss_list.append(loss.item())


        if args.classif:
            Sensitivity,Specifity,g_mean,Precision,Recall,F1,F2 = calculatate_stats(TP,FN,FP,TN)
            print("Test ACC:",acc / 800)
            print("P 1; TP , FN", TP.item(),FN.item())
            print("F 0; FP , TN", FP.item(),TN.item())
        return np.mean(loss_list),acc /800,np.mean(mae_list),0,0, 0, 0,0,0
        #return np.mean(loss_list),acc/800 ,np.mean(mae_list),F1,F2, Precision, Recall,Specifity, g_mean

def calculatate_stats(TP,FN,FP,TN):
    Sensitivity = TP/ (TP + FN)
    Specifity = TN / (TN + FP)
    g_mean = (Specifity *Sensitivity).sqrt()

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    F1 = (2 * Precision * Recall) / (Precision + Recall)
    F2 =(5 * Precision * Recall) / (4 * Precision + Recall)
    if torch.isnan(F1):
        F1 = 0
    if torch.isnan(F2):
        F2 = 0
    if torch.isnan(Precision):
        Precision = 0
    if torch.isnan(Recall):
        Recall = 0
    if torch.isnan(g_mean):
        g_mean = 0
    if torch.isnan(Sensitivity):
        Sensitivity = 0
    if torch.isnan(Specifity):
        Specifity = 0
    return Sensitivity,Specifity,g_mean,Precision,Recall,F1,F2


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
    n_data_points = mu_2d.size()[-1]
    #pdb.set_trace()
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
        idx = torch.where(mask[k] > 0)
        data_masked= data[k][idx]
        mu_masked= mu[k][idx]
        log_prob = gaussian_log_likelihood(mu_masked, data_masked, obsrv_std)
        res.append(log_prob)

    res = torch.stack(res, 0).to(data.device)

    return res

def perturb_TS(data,eps,device,args,abstract_ratio = 0.6,run_backwards = False):
    #returns box of [data,mask] and init data
    assert abstract_ratio <= 1. + 1e-6 and abstract_ratio>= 0.

    if args.dataset == "physionet":
        if abstract_ratio< 1 - 1e-7:

            if not run_backwards:
                ratio = 1  - abstract_ratio
                n_concrete_tp = torch.where(data['observed_tp']<= ratio)[0].shape[0] 

                obs_data_concrete = data["observed_data"][:,:n_concrete_tp,:].clone()
                obs_data_abstract = data["observed_data"][:,n_concrete_tp:,:].clone() 

                obs_data_low, obs_data_upp = obs_data_abstract.clone() - eps, obs_data_abstract.clone()  + eps
                obs_data_low = torch.cat((obs_data_concrete.clone(),obs_data_low),dim=1).to(device)
                obs_data_upp = torch.cat((obs_data_concrete.clone(),obs_data_upp),dim=1).to(device)

                init_low, init_upp = data["observed_init"].clone(), data["observed_init"].clone()
            else:
                n_concrete_tp = torch.where(data['observed_tp']<= abstract_ratio)[0].shape[0] 

                obs_data_abstract = data["observed_data"][:,:n_concrete_tp,:].clone()
                obs_data_concrete = data["observed_data"][:,n_concrete_tp:,:].clone() 

                obs_data_low, obs_data_upp = obs_data_abstract.clone() - eps, obs_data_abstract.clone()  + eps
                obs_data_low = torch.cat((obs_data_low,obs_data_concrete.clone()),dim=1).to(device)
                obs_data_upp = torch.cat((obs_data_upp,obs_data_concrete.clone()),dim=1).to(device)

                init_low, init_upp = data["observed_init"].clone(), data["observed_init"].clone()

        else:
            obs_data_low, obs_data_upp = data["observed_data"].clone() - eps, data["observed_data"].clone() + eps
            init_low, init_upp = data["observed_init"].clone(), data["observed_init"].clone()
            init_low[:,0], init_upp[:,0] = init_low[:,0] - eps , init_upp[:,0] + eps
            init_low[:,2], init_upp[:,2] = init_low[:,2] - eps , init_upp[:,2] + eps

        obs_data_low[torch.where(data["observed_mask"]==0)] *= 0
        obs_data_upp[torch.where(data["observed_mask"]==0)] *= 0
        #clamp 10 and 28
        obs_data_low[:,:,10], obs_data_upp[:,:,10] = torch.clamp(obs_data_low[:,:,10],0.0,1.0),torch.clamp(obs_data_upp[:,:,10],0.0,1.0)
        obs_data_low[:,:,28], obs_data_upp[:,:,28] = torch.clamp(obs_data_low[:,:,28],0.0,1.0),torch.clamp(obs_data_upp[:,:,28],0.0,1.0) 

    obs_data_low,obs_data_upp = torch.cat((obs_data_low,data["observed_mask"]),dim=2).float().to(device),torch.cat((obs_data_upp,data["observed_mask"]),dim=2).float().to(device)
    obs_data_box = HybridZonotope.construct_from_bounds(min_x = obs_data_low, max_x=obs_data_upp, domain="box",dtype = obs_data_upp.dtype)
    init_box = HybridZonotope.construct_from_bounds(min_x = init_low, max_x=init_upp, domain="box",dtype = init_upp.dtype)

    return obs_data_box,init_box

















