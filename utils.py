import numpy as np
import os
import time


import torch as t
from torch import nn as nn
from torch.nn import functional as F
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from model import *
import argparse
import pdb
import torchvision
import torchattacks

from PARC.AIDomains.zonotope import HybridZonotope
from PARC.AIDomains.abstract_layers import Sequential,Linear
from PARC.examples.eps_scheduler import SmoothedScheduler



def save_checkpoint(state, savepath,filename):
    t.save(state,savepath + filename)
    
def check_sound(data,model,input_box,eps, verbose = False):

    model.eval()

    lb, ub = input_box.concretize()

    if verbose:

        print((ub-lb).max().item())

    for _ in range(100):

        random = (2* torch.rand(data.shape)-1)* (eps)

        image = torch.clamp(data.clone() + random.to(data.device),0.0,1.0)

        out = model(image) 

        if not ((lb <= out + 1e-5).all() and (out <= ub + 1e-5).all()):

            print((lb - out).max())

            model.train()

            return False

    model.train()

    return True


def test(model,test_loader,device,args, n_samples= 10000):#works only for MNIST perfectly
    test_acc = 0.0
    num_items = 0
    statistics = []
    model.eval()


    boolean = args.act not in ["baseline","baseline_sin","baseline_sin_10_ps","baseline_sin_10_ns","baseline_10_ps","baseline_10_ns","mini_base","mini_adj"]
    boolean_value = 7
    if args.act == "mini":
        boolean_value = 5

    if boolean and model.net[boolean_value].running_mean_step is not None:
        if model.net[boolean_value].running_mean_step.data == 0.0:
            model.train()
    with t.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader),  total=len(test_loader)):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            if boolean:
                statistics = update_statistics(statistics,model.net[boolean_value].liste)

            test_acc += t.sum(t.argmax(output, dim=1) == target).item()
            num_items += data.shape[0]

            # testing combined with adv. attacks just takes a lifetime therefore reduced to first 512 samples
            if num_items == n_samples or (args.debug in [1,2] and num_items == 50):
                break
    if boolean:
        if len(statistics)> 5:
            statistics[6] = [item for item in statistics[6] if np.isnan(item) == False]
            statistics[7] = [item for item in statistics[7] if np.isnan(item) == False]
            if len(statistics[6]) == 0:
                statistics[6] = [0]
            if len(statistics[7]) == 0:
                statistics[7] = [0]
        return test_acc / num_items, [np.round(np.mean(statistics[i]),4)  for i in range(len(statistics))]
    return test_acc / num_items,[[1],[1],[1]]

def update_statistics(stats,liste):
    if len(stats) == 0:
        stats = [[] for i in range(len(liste))]
    for i in range(len(liste)):
        stats[i] += [liste[i]]
    return stats

def train(model,criterion,train_loader,optimizer,device,args):
    train_losses = []
    statistics = []
    accuracy = 0.0
    num_items = 0
    model.train()
    boolean = args.act not in ["baseline","baseline_sin","baseline_sin_10_ps","baseline_sin_10_ns","baseline_10_ps","baseline_10_ns","mini_base","mini_adj"]

    boolean_value = 7
    if args.act == "mini":
        boolean_value = 5

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.cuda()
        target=target.cuda()
        output = model(data)
        if boolean:
            try:
                statistics = update_statistics(statistics,model.net[boolean_value].liste)
            except:
                statistics = update_statistics(statistics,model[boolean_value].liste)
        loss = criterion(output, target) 
        if args.adjoint == 3:
            loss += model.net[boolean_value].ss_loss *args.ss_loss
        loss.backward()
        if (batch_idx +1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_losses += [loss.item()]
        accuracy += t.sum(t.argmax(output, dim=1) == target).item()
        num_items += data.shape[0]
        del loss, output   #I DONT like those two lines, but otherwise memory usage increases steadily in NODE models
        torch.cuda.empty_cache()
        if batch_idx == 9 and args.debug == 1:
            break

    if boolean:
        if len(statistics)> 5:
            statistics[6] = [item for item in statistics[6] if np.isnan(item) == False]
            statistics[7] = [item for item in statistics[7] if np.isnan(item) == False]
            if len(statistics[6]) == 0:
                statistics[6] = [0]
            if len(statistics[7]) == 0:
                statistics[7] = [0]
        return np.mean(train_losses), accuracy/ num_items, [np.round(np.mean(statistics[i]),4) for i in range(len(statistics))]
    return np.mean(train_losses), accuracy/ num_items,[[1],[1],[1]]


def train_adv(model,criterion,train_loader,optimizer,device,adv,args):
    train_losses = []
    accuracy = 0.0
    num_items = 0
    statistics = []
    boolean = args.act not in ["baseline","baseline_sin","baseline_sin_10_ps","baseline_sin_10_ns","baseline_10_ps","baseline_10_ns","mini_base"]

    boolean_value = 7
    if args.act == "mini":
        boolean_value = 5

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        model.eval()
        data = data.to(device)
        target=target.to(device)
        adv_data = adv(data,target)
        data.detach()

        model.train()  
        output = model(adv_data)
        if boolean:
            try:
                statistics = update_statistics(statistics,model.net[boolean_value].liste)
            except:
                statistics = update_statistics(statistics,model[boolean_value].liste)

        loss = criterion(output, target) 
        if args.adjoint == 3:
            loss += model.net[7].ss_loss *args.ss_loss
        loss.backward()
        if (batch_idx +1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_losses += [loss.item()]
        accuracy += t.sum(t.argmax(output, dim=1) == target).item()
        num_items += data.shape[0]
        del loss, output   #I DONT like those two lines, but otherwise memory usage increases steadily
        torch.cuda.empty_cache()
        
    if boolean:
        if len(statistics)> 5:
            statistics[6] = [item for item in statistics[6] if np.isnan(item) == False]
            statistics[7] = [item for item in statistics[7] if np.isnan(item) == False]
            if len(statistics[6]) == 0:
                statistics[6] = [0]
            if len(statistics[7]) == 0:
                statistics[7] = [0]
        return np.mean(train_losses), accuracy/ num_items, [np.round(np.mean(statistics[i]),4) for i in range(len(statistics))]
    return np.mean(train_losses), accuracy/ num_items,[[1],[1],[1]]


def train_box_provable(box_model,criterion,train_loader,optimizer,eps,device,args,classes = 10):
    train_losses = []
    accuracy = 0.0
    num_items = 0
    statistics = []
    interval_length = []
    numb_of_t = []
    numb_of_splits = []
    boolean = args.act not in ["baseline","baseline_sin","baseline_sin_10_ps","baseline_sin_10_ns","baseline_10_ps","baseline_10_ns","mini_base"]

    boolean_value = 7
    if args.act == "mini":
        boolean_value = 5

    final_abstract_layers = []
    for i in range(classes):
        eye = torch.eye(classes).to(device).type(box_model[1].weight.type())
        eye[i,:] -= 1
        final_lin_layer =  nn.Linear(classes,classes,bias = None).to(device)
        final_lin_layer.weight.data = eye.T
        final_abstract_layers.append(Linear.from_concrete_layer(final_lin_layer,box_model[-1].output_dim))

    mse = nn.MSELoss()
    box_model.train()
    lamda = eps / (args.target)

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.to(device)
        target=target.to(device)

        concrete_output = box_model.forward(data)
        concrete_loss = criterion(concrete_output,target)
        accuracy += t.sum(t.argmax(concrete_output, dim=1) == target).item()

        if boolean:
            statistics = update_statistics(statistics,box_model[boolean_value].liste)
        input_box = HybridZonotope.construct_from_noise(x=data.clone(), eps=eps, domain="box", data_range=(0.0, 1.0))

        abstract_output = box_model.forward_between(0,len(box_model.layers)-1,input_box)
        abstract_output = HybridZonotope.construct_from_bounds(abstract_output.head - abstract_output.beta,abstract_output.head + abstract_output.beta, domain="zono")
        abstract_output = box_model.forward_between(len(box_model.layers)-1,len(box_model.layers),abstract_output)

        number_of_trajectories = int(abstract_output.shape[0]/ data.shape[0])
        numb_of_t += [number_of_trajectories]
        target = target.repeat(number_of_trajectories)
        
        abstract_loss = 0
        box_loss = 0
        for i in range(classes):
            idx = torch.where(target == i)
            if idx[0].shape[0] == 0:
                continue
            out = final_abstract_layers[i](abstract_output[idx])
            temp_lb,temp_ub = out.concretize() 
            interval_length += [(temp_ub -temp_lb).mean().item()]
            box_loss += mse(temp_ub,temp_lb)
            out = out.get_wc_logits(target[idx])
            abstract_loss += criterion(out,target[idx]) * idx[0].shape[0]
        abstract_loss /= target.shape[0]
        box_loss /= target.shape[0]


        if "dopri5" in args.method:
            numb_of_splits += [box_model[5].splits]
            loss = (1-lamda * 0.66) * concrete_loss + lamda * 0.66 *abstract_loss  + args.gamma * box_loss + box_model[5].error_penalty *args.omega
        else:
            numb_of_splits += [0]
            loss = (1-lamda * args.delta) * concrete_loss + lamda * args.delta *abstract_loss  + args.gamma * box_loss


        loss.backward()
        torch.nn.utils.clip_grad_norm_(box_model.parameters(), args.clip)
        
        if (batch_idx +1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_losses += [loss.item()]
        
        num_items += data.shape[0]
        del loss, concrete_output,abstract_output   #I DONT like those two lines, but otherwise memory usage increases steadily in NODE models
        torch.cuda.empty_cache()
        if batch_idx == 9 and args.debug == 1:
            break
        if batch_idx == 39 and args.debug == 2:
            break

    print("MEAN INTERVAL LENGTH:           ",np.mean((interval_length)))
    print("MEAN NUMBER OF TRAJECTORIES:    ",np.mean(numb_of_t))
    print("MEAN NUMBER OF SPLITS:          ",np.mean(numb_of_splits))
    try:
        box_model[5].update_probabilities()
        print("CURRENT RUNNING MEAN:           ",box_model[5].running_mean_step.item())
    except:
        print("Fixed step")
    if boolean:
        if len(statistics)> 5:
            statistics[6] = [item for item in statistics[6] if np.isnan(item) == False]
            statistics[7] = [item for item in statistics[7] if np.isnan(item) == False]
            if len(statistics[6]) == 0:
                statistics[6] = [0]
            if len(statistics[7]) == 0:
                statistics[7] = [0]
        return np.mean(train_losses), accuracy/ num_items, [np.round(np.mean(statistics[i]),4) for i in range(len(statistics))],np.mean((interval_length)),np.mean(numb_of_t),np.mean(numb_of_splits)
    return np.mean(train_losses), accuracy/ num_items,1,1,1

def load_dataset(args):
    if args.dataset == "MNIST":
        #MNIST DATASET
        norm_mean,norm_std = [0.1307],[0.3081]
        if args.debug in [0]:
            train_loader = t.utils.data.DataLoader(torchvision.datasets.MNIST("data/mnist", train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()])),batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = t.utils.data.DataLoader(torchvision.datasets.MNIST("data/mnist", train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()])),batch_size=args.batch_size, shuffle=False)

        if args.adv in ["box","box_lin","lin","box_trades"]:
            test_loader = t.utils.data.DataLoader(
                torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=1, shuffle=False)
        else:
            test_loader = t.utils.data.DataLoader(
                torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=args.batch_size, shuffle=False)

    elif args.dataset == "F-MNIST":
        #FMNIST DATASET
        norm_mean,norm_std = [0.2860], [0.3530]
        if args.debug in [0]:
            train_loader = t.utils.data.DataLoader(torchvision.datasets.FashionMNIST("data/f-mnist", train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()])),batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = t.utils.data.DataLoader(torchvision.datasets.FashionMNIST("data/f-mnist", train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()])),batch_size=args.batch_size, shuffle=False)

        if args.adv in ["box","box_lin","lin","box_trades"]:
            test_loader = t.utils.data.DataLoader(
                torchvision.datasets.FashionMNIST("data/f-mnist", train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=1, shuffle=False)
        else:
            test_loader = t.utils.data.DataLoader(
                torchvision.datasets.FashionMNIST("data/f-mnist", train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])), batch_size=args.batch_size, shuffle=False)
    elif args.dataset == "CIFAR10":
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2023, 0.1994, 0.2010]

        train_loader = t.utils.data.DataLoader(torchvision.datasets.CIFAR10("data/cifar10",train = True,download=True,transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()])),batch_size= args.batch_size,shuffle = True)
        #maybe val set
        if args.adv in ["box","box_lin","lin","box_trades"]:

            test_loader = t.utils.data.DataLoader(torchvision.datasets.CIFAR10("data/cifar10",train = False,download=True,transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()])),batch_size= 1,shuffle = False)
        else:
            test_loader = t.utils.data.DataLoader(torchvision.datasets.CIFAR10("data/cifar10",train = False,download=True,transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()])),batch_size= args.batch_size,shuffle = False)

    else:
        print("NOT IMPLEMENTED DATASET")
        exit()

    return train_loader,test_loader,norm_mean,norm_std


