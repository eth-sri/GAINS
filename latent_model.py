import numpy as np
import os

import torch 
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import argparse
import pdb
import torchdiffeq._impl.odeint as odeint


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents, 
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.input_dim = input_dim
        self.activations = []
        if method == "dopri5": #just a quick hack on how to deal with different tolarance values
            self.met = method
            self.ode_method = "dopri5"
            self.atol = 0.005
            self.rtol = 0.0
            self.adaptive_step_factor = None
            self.running_mean_step = None
            self.anode = 0
            self.step_size = None

        elif method == "dopri5_0.005": #just a quick hack on how to deal with different tolarance values
            self.met = method
            self.ode_method = "dopri5"
            self.atol = 0.005
            self.rtol = 0.0
            self.adaptive_step_factor = None
            self.running_mean_step = None
            self.anode = 0
            self.step_size = None

        elif method == "dopri5_0.00005": #just a quick hack on how to deal with different tolarance values
            self.met = method
            self.ode_method = "dopri5"
            self.atol = 0.00005
            self.rtol = 0.0
            self.adaptive_step_factor = None
            self.running_mean_step = None
            self.anode = 0
            self.step_size = None


        elif method == "dopri5_0.005_2a":
            self.met = method
            self.ode_method = "dopri5"
            self.atol = 0.005
            self.rtol = 0.0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.step_size = None
        elif method == "dopri5_0.01_2a":
            self.met = method
            self.ode_method = "dopri5"
            self.atol = 0.01
            self.rtol = 0.0
            self.adaptive_step_factor = 2
            self.running_mean_step = nn.parameter.Parameter(torch.zeros(1),requires_grad= False)
            self.step_size = None

        elif method == "euler":
            self.met = method
            self.ode_method = "euler"
            self.atol = 0.0
            self.rtol = 0.0
            self.adaptive_step_factor = None
            self.running_mean_step = None
            self.anode = 0
            self.step_size = None

        elif method == "euler_2":
            self.met = method
            self.ode_method = "euler"
            self.atol = 0.0
            self.rtol = 0.0
            self.adaptive_step_factor = None
            self.running_mean_step = None
            self.step_size = None

        self.latents = latents      
        self.device = device
        self.ode_func = ode_func
        self.odesolver = odeint

        self.odeint_rtol = self.rtol
        self.odeint_atol = self.atol

    def forward(self, first_point, time_steps_to_predict, backwards = False,mask = None):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]


        if self.met in ["dopri5", "dopri5_0.00005","dopri5_0.005"]:
            if 0 not in time_steps_to_predict:
                time_steps_to_predict = torch.cat((torch.zeros_like(time_steps_to_predict[0:1]),time_steps_to_predict),0)
                temp_flag = True
            else:
                temp_flag = False
            pred_y = self.odesolver(self.ode_func, first_point, time_steps_to_predict, 
                rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
            self.liste = pred_y[1]
            pred_y = pred_y[0].permute(2,0,1,3)
            if temp_flag:
                pred_y = pred_y[:,1:]

        elif self.met == "euler":
            #error shape no influence
            pred_y = self.odesolver(self.ode_func, first_point, time_steps_to_predict, 
                rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
            pred_y = pred_y[0].permute(1,2,0,3)

        elif self.met == "euler_2":    
            out_data = torch.zeros_like(first_point).unsqueeze(2).repeat([1,1,len(time_steps_to_predict),1])
            step_sizes = []
            for i in range(len(time_steps_to_predict)):
                idx = torch.where(mask[:,i].sum(1) > 0)[0]
                inputs = first_point[0:1,idx]

                if time_steps_to_predict[i] == 0:
                    out_data[0:1,idx,i] += inputs
                    continue
                to_predict = torch.tensor([0,time_steps_to_predict[i]]).to(time_steps_to_predict.device).type(time_steps_to_predict.dtype)
                self.step_size = time_steps_to_predict[i] / 2
                out, self.liste = self.odesolver(self.ode_func, inputs, to_predict, method = self.ode_method,
                                    rtol= self.odeint_rtol,atol= self.odeint_atol,step_size= self.step_size,
                                    adaptive_step_factor = self.adaptive_step_factor, running_mean_step = None) 

                out_data[0:1,idx,i] += out[1]

            pred_y = out_data.permute(1,2,0,3)

        elif self.met in ["dopri5_0.005_2a","dopri5_0.01_2a"]:    
            out_data = torch.zeros_like(first_point).unsqueeze(2).repeat([1,1,len(time_steps_to_predict),1])
            step_sizes = []
            trajectories = []
            for i in range(len(time_steps_to_predict)):
                idx = torch.where(mask[:,i].sum(1) > 0)[0]
                
                inputs = first_point[0,idx]

                if time_steps_to_predict[i] == 0:
                    out_data[0,idx,i] += inputs
                    continue
                to_predict = torch.tensor([0,time_steps_to_predict[i]]).to(time_steps_to_predict.device).type(time_steps_to_predict.dtype)
                if self.training:
                    out, self.liste = self.odesolver(self.ode_func, inputs, to_predict, method = self.ode_method,
                                        rtol= self.odeint_rtol,atol= self.odeint_atol,step_size= self.step_size,
                                        adaptive_step_factor = self.adaptive_step_factor, running_mean_step = None) 
                    step_sizes.append(self.liste[-2])
                    trajectories.append(self.liste[-4])
                else:
                    out,self.liste = self.odesolver(self.ode_func, inputs, to_predict, method = self.ode_method,
                        rtol= self.odeint_rtol,atol= self.odeint_atol,step_size= self.step_size,
                        adaptive_step_factor = self.adaptive_step_factor,running_mean_step = torch.clamp(self.running_mean_step.data,0.0,to_predict[-1]))
                out_data[0,idx,i] += out[1]

            self.step_sizes = step_sizes
            self.trajectories = trajectories
            if self.training:
                self.running_mean_step.data = torch.clamp((1-0.1)* self.running_mean_step.data + 0.1 * min(step_sizes),0,1)

            pred_y = out_data.permute(1,2,0,3)
        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
        n_traj_samples = 1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y

class LatentODE(nn.Module):
    def __init__(self,args,input_dim,input_init,device,n_labels=1):
        super(LatentODE,self).__init__()
        dim = args.latents
        if args.act == "relu6":
            nonlin = nn.ReLU6
        elif args.act == "tanh":
            nonlin = nn.Tanh
        elif args.act == "relu":
            nonlin = nn.ReLU
        else:
            nonlin = nn.ReLU
        self.device = device
        self.z0_prior = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
        self.sample_normal = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))

        self.ode_func_gen = _ODEfunc(dim, args.latents,n_layers = args.gen_layers, n_units = args.units, nonlinear = nonlin).to(device) 

        n_rec_dims = args.rec_dims
        enc_input_dim = int(input_dim) * 2 # we concatenate the mask
        gen_data_dim = int(input_dim)
        z0_dim = args.latents
        self.z0_dim = z0_dim

        self.ode_func_rec = _ODEfunc(n_rec_dims, n_rec_dims, n_layers = args.rec_layers, n_units = args.units, nonlinear = nonlin).to(device)

        self.z0_diffeq_solver = DiffeqSolver(enc_input_dim, self.ode_func_rec, "euler", args.latents, 
            odeint_rtol = 0, odeint_atol = 0, device = device)

        self.encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, self.z0_diffeq_solver,input_init = input_init,
            z0_dim = z0_dim, n_gru_units = args.gru_units, device = device,nonlin = nonlin).to(device)

        self.decoder = nn.Sequential(nn.Linear(args.latents, gen_data_dim),).to(device)

        self.diffeq_solver = DiffeqSolver(gen_data_dim, self.ode_func_gen, args.method, args.latents, 
            odeint_rtol = 0, odeint_atol = 5e-3, device = device)

        self.classif = args.classif
        classif_dim = 50
        self.classifier = nn.Sequential(nn.Linear(z0_dim, classif_dim),
            nn.ReLU(),
            nn.Linear(classif_dim, classif_dim),
            nn.ReLU(),
            nn.Linear(classif_dim, n_labels),).to(device)

    def forward(self,x,time_steps_observed,time_steps_to_predict,init_data,run_backwards = False,only_classif = False,target_mask = None):
        #assumes x = [x,mask]
        first_point_mu, first_point_std = self.encoder_z0(x,time_steps_observed, init_data,run_backwards = run_backwards)
        if self.training:
                first_point_enc = self.sample_normal.sample(first_point_mu.size()).squeeze(-1) * first_point_std + first_point_mu
        else:
            first_point_enc = first_point_mu

        if only_classif:
            all_extra_info = {"label_predictions":self.classifier(first_point_enc).squeeze(-1)}
            return all_extra_info


        if self.diffeq_solver.met in ["dopri5_0.005_2a","dopri5_0.01_2a","euler_2"]:
            sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict,mask=target_mask)
        else:
            sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }
        if self.classif:
            all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)
        return pred_x,all_extra_info

class Baseline_RNN(nn.Module):
    def __init__(self,args,input_dim,input_init,device,n_labels=1):
        super(Baseline_RNN,self).__init__()
        dim = args.latents
        if args.act == "relu6":
            nonlin = nn.ReLU6
        elif args.act == "tanh":
            nonlin = nn.Tanh
        elif args.act == "relu":
            nonlin = nn.ReLU
        else:
            nonlin = nn.ReLU
        self.device = device
        self.z0_prior = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
        self.sample_normal = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))

        self.ode_func_gen = _ODEfunc(dim, args.latents,n_layers = args.gen_layers, n_units = args.units, nonlinear = nonlin).to(device) 

        n_rec_dims = args.rec_dims
        enc_input_dim = int(input_dim) * 2 # we concatenate the mask
        gen_data_dim = int(input_dim)
        z0_dim = args.latents
        self.z0_dim = z0_dim

        self.encoder_z0 = Encoder_z0_RNN(n_rec_dims, enc_input_dim, input_init = input_init,
                z0_dim = z0_dim, n_gru_units = args.gru_units, device = device).to(device)

        self.decoder = nn.Sequential(nn.Linear(args.latents, gen_data_dim),).to(device)

        self.ode_func_gen = _ODEfunc(dim, args.latents,n_layers = args.gen_layers, n_units = args.units, nonlinear = nonlin).to(device) 

        self.resnet = DiffeqSolver_Resnet(gen_data_dim, self.ode_func_gen, args.latents, device = device)

        self.classif = args.classif
        classif_dim = 50
        self.classifier = nn.Sequential(nn.Linear(z0_dim, classif_dim),
            nn.ReLU(),
            nn.Linear(classif_dim, classif_dim),
            nn.ReLU(),
            nn.Linear(classif_dim, n_labels),).to(device)
    
    def forward(self,x,time_steps_observed,time_steps_to_predict,init_data,run_backwards = False,only_classif = False,target_mask = None):
        #assumes x = [x,mask]
        first_point_mu, first_point_std = self.encoder_z0(x,time_steps_observed, init_data,run_backwards = run_backwards)
        if self.training:
                first_point_enc = self.sample_normal.sample(first_point_mu.size()).squeeze(-1) * first_point_std + first_point_mu
        else:
            first_point_enc = first_point_mu


        sol_y = self.resnet(first_point_enc, time_steps_to_predict,mask=target_mask)
        pred_x = self.decoder(sol_y)
        

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }
        if self.classif:
            all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)
        return pred_x,all_extra_info


class _ODEfunc(nn.Module):
    def __init__(self,n_inputs, n_outputs, n_layers = 1, n_units = 100, nonlinear = nn.ReLU):
        super(_ODEfunc, self).__init__()
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, t, x, backwards = False):
        grad = self.layers(x)
        if backwards:
            grad = -grad
        return grad


class Encoder_z0_ODE_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None, 
        z0_dim = None, GRU_update = None, 
        n_gru_units = 100, 
        device = torch.device("cpu"),input_init= 7,nonlin=nn.ReLU):
        
        super(Encoder_z0_ODE_RNN, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device,nonlin = nonlin).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.input_init = input_init
        self.device = device
        self.extra_info = None

        if input_init >0:
            self.hidden_init = nn.Sequential(nn.Linear(input_init,2*latent_dim),nonlin()).to(device)
        else:
            self.hidden_init = None

        if input_init == 0:
            inbet_dim = 100
        else:
            inbet_dim = 100
        self.transform_z0 = nn.Sequential(
           nn.Linear(latent_dim * 2, inbet_dim),
           nn.ReLU(),
           nn.Linear(inbet_dim, self.z0_dim * 2),).to(device)
        #init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, init_data,run_backwards = False, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        n_traj, n_tp, n_dims = data.size()

        if len(time_steps) == 1:

            if (data.type() == 'torch.DoubleTensor' or data.type() == 'torch.cuda.DoubleTensor') and self.hidden_init != None:
                temp = self.hidden_init(init_data.double())

            elif self.hidden_init != None:
                temp = self.hidden_init(init_data.float())
            else:
                temp = torch.zeros((data.shape[0],2* self.latent_dim)).to(data.device)
            

            prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
            prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps,init_data, run_backwards = run_backwards,
                save_info = save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps,init_data,
        run_backwards = False, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        if run_backwards:
            t0 = time_steps[-1]
            prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]
        else:
            t0 = time_steps[0]
            prev_t, t_i = time_steps[0] * 0,  time_steps[0]


        if (data.type() == 'torch.DoubleTensor' or data.type() =="torch.cuda.DoubleTensor") and self.hidden_init != None:

            temp = self.hidden_init(init_data.double())
        elif self.hidden_init != None:
            temp = self.hidden_init(init_data.float())
        else:
            temp = torch.zeros((data.shape[0],2* self.latent_dim)).to(data.device)

        prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
        prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length 

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if (prev_t - t_i).abs() <1e-8:
                yi_ode = prev_y
            else:
                if (prev_t - t_i).abs() < minimum_step:
                    time_points = torch.stack((prev_t, t_i)).to(self.device)
                else:
                    n_intermediate_tp = max(2, ((prev_t - t_i).abs() / minimum_step).int())
                    #do multiple steps for better accuracy, but don't do to small steps
                    time_points = torch.linspace(prev_t, t_i, n_intermediate_tp).to(self.device)
                        
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)
                assert(not torch.isnan(ode_sol).any())
                yi_ode = ode_sol[:, :, -1, :]

            xi = data[:,i,:].unsqueeze(0)
            
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            prev_y, prev_std = yi, yi_std     

            if run_backwards:
                prev_t, t_i = time_steps[i],  time_steps[i-1]
            elif i < (len(time_steps)-1):
                prev_t,t_i =  time_steps[i], time_steps[i+1]

            latent_ys.append(yi)

            if save_info:
                d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
                     "yi": yi.detach(), "yi_std": yi_std.detach(), 
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        assert(not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys, extra_info

class Encoder_z0_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, 
        z0_dim = None, GRU_update = None, 
        n_gru_units = 100, 
        device = torch.device("cpu"),input_init= 7):
        
        super(Encoder_z0_RNN, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device,RNN_enc = True).to(device)
        else:
            self.GRU_update = GRU_update

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.input_init = input_init
        self.device = device
        self.extra_info = None

        self.hidden_init = nn.Sequential(nn.Linear(input_init,2*latent_dim),nn.ReLU()).to(device)

        if input_init == 0:
            inbet_dim = 2* latent_dim
        else:
            inbet_dim = 100
        self.transform_z0 = nn.Sequential(
           nn.Linear(latent_dim * 2, inbet_dim),
           nn.ReLU(),
           nn.Linear(inbet_dim, self.z0_dim * 2),).to(device)
        #init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, init_data,run_backwards = False, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()

        if len(time_steps) == 1:
            if data.type() == 'torch.DoubleTensor' or data.type() == 'torch.cuda.DoubleTensor':
                temp = self.hidden_init(init_data.double())

            else:
                temp = self.hidden_init(init_data.float())
            

            prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
            prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps,init_data, run_backwards = run_backwards,
                save_info = save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps,init_data,
        run_backwards = False, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        if run_backwards:
            t0 = time_steps[-1]
            prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]
        else:
            t0 = time_steps[0]
            prev_t, t_i = time_steps[0] * 0,  time_steps[0]

        if data.type() == 'torch.DoubleTensor' or data.type() == 'torch.cuda.DoubleTensor':
            temp = self.hidden_init(init_data.double())
        else:
            temp = self.hidden_init(init_data.float())
        prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
        prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length 

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:

            xi = data[:,i,:].unsqueeze(0)
            prev_y, prev_std = self.GRU_update(prev_y ,prev_std, xi)   

            if run_backwards:
                prev_t, t_i = time_steps[i],  time_steps[i-1]
            elif i < (len(time_steps)-1):
                prev_t,t_i =  time_steps[i], time_steps[i+1]

            latent_ys.append(prev_y)


        latent_ys = torch.stack(latent_ys, 1)

        return prev_y, prev_std, latent_ys, extra_info

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res
def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, 
        update_gate = None,
        reset_gate = None,
        new_state_net = None,
        n_units = 100,
        device = torch.device("cpu"),
        nonlin = nn.ReLU,
        RNN_enc = False):
        super(GRU_unit, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.RNN_enc = RNN_enc
        if update_gate is None:
            self.update_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim // 2, n_units),
               nonlin(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
            #utils.init_network_weights(self.update_gate)
        else: 
            self.update_gate  = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim // 2, n_units),
               nonlin(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
            #utils.init_network_weights(self.reset_gate)
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim // 2, n_units),
               nonlin(),
               nn.Linear(n_units, latent_dim * 2))
            #utils.init_network_weights(self.new_state_net)
        else: 
            self.new_state_net  = new_state_net


    def forward(self, y_mean, y_std, x, masked_update = True):
        x,mask = split_last_dim(x)
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
        
        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std
        
        if masked_update and not self.RNN_enc:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            assert(torch.sum(x[mask == 0.] != 0.) == 0)
            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std

        assert (new_y_std>= 0).all()
        return new_y, new_y_std


