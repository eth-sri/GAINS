import torch
import torch.nn as nn
import pdb
import torchdiffeq._impl.odeint as odeint

class Bias(nn.Module):
    def __init__(self, in_dim=None, bias=None):
        super().__init__()
        assert in_dim is not None or bias is not None
        in_dim = list(bias.shape) if in_dim is None else in_dim
        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]
        if bias is not None:
            self.bias = bias
        else:
            self.bias = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return x + self.bias
        
class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)
        
    def forward(self, t, x):
        return self._layer(torch.cat([torch.ones_like(x[:, :1, :, :]) * t, x], 1))

class Scale(nn.Module):
    def __init__(self, in_dim=None, scale=None):
        super().__init__()
        assert in_dim is not None
        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]
        if scale is not None:
            self.scale = scale
        else:
            self.scale = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        return x * self.scale

class DeNormalization(nn.Module):
    def __init__(self, in_dim=None, mean=None, std=None):
        super().__init__()
        assert in_dim is not None
        self.mean = torch.nn.Parameter(torch.tensor(0.) if mean is None else torch.tensor(mean, dtype=torch.float), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor(0.) if std is None else torch.tensor(std, dtype=torch.float), requires_grad=False)

        if len(in_dim) in [3, 4]:
            self.mean.data = self.mean.data.view(1, -1, 1, 1)
            self.std.data = self.std.data.view(1, -1, 1, 1)
        elif len(in_dim) in [1, 2]:
            self.mean.data = self.mean.data.view(1, -1)
            self.std.data = self.std.data.view(1, -1)
        else:
            assert False

        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]

    def forward(self, x):
        return x * self.std + self.mean

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)
        
    def forward(self, t, x):
        return self._layer(torch.cat([torch.ones_like(x[:, :1, :, :]) * t, x], 1))

class ODEBlock(nn.Module):
    def __init__(self, odefunc, met= "dopri5" ,adjoint = 0 ,t=[0,1]): #maybe try different time
        #odefunc is an instance of ODEfunc, which contains the "mini-network" that describes the ODE function
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor(t).float()
        self.met = met
        self.method = met
        self.atol = 1e-9 #default value
        self.rtol= 1e-7  #default value
        self.adjoint = adjoint
        self.step_size = 0.1 # default value

        if adjoint == 0: #store forward passes
            self.odesolver = odeint
            self.anode = 0
        if adjoint == 3: #store forward passes
            self.odesolver = odeint
            self.anode = 3
        elif adjoint == 1: #adjoint method
            self.odesolver = odeint_adjoint
            self.anode = 1
        elif adjoint == 2: #ANODE checkpointing same as 0 if only one ODeblock
            self.odesolver = odeint_anode
            self.options = {}
            self.options.update({'Nt':int(2)})
            self.options.update({'method':met})
            self.anode = 2

        if met == "dopri5_0.1": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.1
            self.rtol = 0.1
        if met == "dopri5_0.01": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.01
            self.rtol = 0.01
        if met == "dopri5_0.001": #just a quick hack on how to deal with different tolarance values
            self.met = "dopri5"
            self.atol = 0.001
            self.rtol = 0.001
        if met == "rk4_10": #just a quick hack on how to deal with different tolarance values
            self.met = "rk4"
            self.step_size = t[1]/10
        if met == "rk4_0.1": 
            self.met = "rk4"
            self.step_size = 0.1
        if met == "rk4_2": 
            self.met = "rk4"
            self.step_size = t[1]/2
        if met == "rk4_5":
            self.met = "rk4"
            self.step_size = t[1]/5

        if met == "euler_2": 
            self.met = "euler"
            self.step_size = t[1]/2
        if met == "euler_5": 
            self.met = "euler"
            self.step_size = t[1]/5
        if met == "euler_10": 
            self.met = "euler"
            self.step_size = t[1]/10

        if met == "mp_2": 
            self.met = "midpoint"
            self.step_size = t[1]/2
        if met == "mp_5": 
            self.met = "midpoint"
            self.step_size = t[1]/5
        if met == "mp_10": 
            self.met = "midpoint"
            self.step_size = t[1]/10


    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)

        if self.anode == 2:
            #pdb.set_trace()
            out = self.odesolver(self.odefunc, x, options = self.options)
            return out
        elif self.anode == 3:
            out,self.n_tot, self.n_acc, self.avg_step, self.ss_loss= self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,steady_state = 0)
            self.ss_loss = (self.ss_loss.flatten(1)).pow(2).sum(1).sqrt().mean()
            return out[1]
        else:
            out,self.n_tot, self.n_acc, self.avg_step = self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size) 
            return out[1] # Two time points are stored in [0,endtime], here the state at time 1 is output.
    def forward_ss(self,x):
        out,self.n_tot, self.n_acc, self.avg_step, self.ss_loss= self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,steady_state = 0)
        self.ss_loss = (self.ss_loss.flatten(1)).pow(2).sum(1).sqrt().mean()
        return out[1]

    def forward_path(self,x):
        out,self.n_tot, self.n_acc, self.avg_step, self.ss_loss= self.odesolver(self.odefunc, x, self.integration_time, method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,steady_state = 1)
        self.ss_loss = (self.ss_loss.flatten(1)).pow(2).sum(1).sqrt().mean()
        return out[1]
    
    def set_solver(self,method):
        self.method = method
        if method == 'rk4_10':
            self.met = 'rk4'
            self.step_size= self.integration_time[-1]/10
        elif method == 'rk4_0.1':
            self.met = 'rk4'
            self.step_size= 0.1
        elif method == 'dopri5_0.1':
            self.met = 'dopri5'
            self.rtol = 0.1
            self.atol = 0.1
        elif method == 'dopri5':
            self.met = 'dopri5'
            self.rtol = 1e-7
            self.atol = 1e-9
        elif method == 'dopri5_0.001':
            self.met = 'dopri5'
            self.rtol = 0.001
            self.atol = 0.001
        else:
            print("Not implemented yet:",method)
            exit()


           
class ODEfunc(nn.Module):
    def __init__(self, in_dim,hid_dim, act = "relu",nogr = 32):
        super(ODEfunc, self).__init__()
        self.in_dim = in_dim    #I generally allow different numbers of inbetween "channels"
        self.hid_dim = hid_dim

        if act == "relu":
            self.act1 = nn.ReLU(inplace = True)
            self.act2 = nn.ReLU(inplace = True)
        elif act == "elu":
            self.act1 = nn.ELU(inplace=True)
            self.act2 = nn.ELU(inplace=True)
        elif act == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(inplace=True)
        elif act == "sin":
            self.act1 = torch.sin
            self.act2 = torch.sin
        elif act == "cos":
            self.act = torch.cos
        else:
            print("Choosen activation:",act, "not implemented yet.")
            exit()

        self.conv1 = ConcatConv2d(in_dim, hid_dim, 3, 1, 1)
        self.norm1 = norm(hid_dim,nogr)
        #self.norm1 = norm_LN(hid_dim)
        self.conv2 = ConcatConv2d(hid_dim, in_dim, 3, 1, 1)
        self.norm2 = norm(in_dim,nogr)
        #self.norm2 = norm_LN(in_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1 #count the number of forward passes
        out = self.conv1(t, x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(t, out)
        out = self.norm2(out)
        out = self.act2(out)
        return out

class Normalization(nn.Module):
    #Normalization layer 
    def __init__(self,device, mean,std):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([mean]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([std]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma

def norm(dim,nogr=32):
    return nn.GroupNorm(min(nogr, dim), dim)

class BatchNorm(nn.Module):
    def __init__(self,num_features,device,n_max_tp,endtime):
        super(BatchNorm, self).__init__()

        self.n_max_tp = n_max_tp
        self.device = device
        self.endtime = endtime
        self.num_features = num_features
        
        if n_max_tp % 2 == 1:
            equi_dist = endtime / (n_max_tp - 1)
        else:
            equi_dist = endtime / n_max_tp 
        #init BN-layer statistics at equidistant timepoint
        self.timepoints = torch.zeros(n_max_tp).to(device)
        seq = []
        for i in range(n_max_tp):
            self.timepoints[i] = i * equi_dist
            seq.append(nn.BatchNorm2d(num_features).to(device))
        self.BatchNorm2d = nn.Sequential(*seq)

    def forward(self,t, x):
        key= (self.timepoints - t).abs().argmin()
        out= self.BatchNorm2d[key](x)
        return out

class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, 
        update_gate = None,
        reset_gate = None,
        new_state_net = None,
        n_units = 100,
        device = torch.device("cpu")):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.ReLU(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
        else: 
            self.update_gate  = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.ReLU(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.ReLU(),
               nn.Linear(n_units, latent_dim * 2))
        else: 
            self.new_state_net  = new_state_net


    def forward(self, y_mean, y_std, x, masked_update = True):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
        
        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            n_data_dims = x.size(-1)//2
            mask = x[:, :, n_data_dims:]
            assert(torch.sum(x[:, :, :n_data_dims][mask == 0.] != 0.) == 0)
            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std

        assert (new_y_std>= 0).all()
        return new_y, new_y_std
class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim,  
        z0_dim = None,
        device = torch.device("cpu"),input_init= 7,z0_diffeq_solver = None, GRU_update = None, n_gru_units = 100 ):
        
        super(Encoder_z0_ODE_RNN, self).__init__()
        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.input_init = input_init
        self.device = device


        self.hidden_init = nn.Sequential(nn.Linear(input_init,2*latent_dim),nn.ReLU()).to(device)
        self.transform_z0 = nn.Sequential(
           nn.Linear(latent_dim * 2, 100),
           nn.ReLU(),
           nn.Linear(100, self.z0_dim * 2),).to(device)


    def forward(self, data, time_steps, init_data,run_backwards = False, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            temp = self.hidden_init(init_data.float())
            prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
            prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            last_yi, last_yi_std = self.run_odernn(
                data, time_steps,init_data, run_backwards = run_backwards,
                save_info = save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = split_last_dim(self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
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

        temp = self.hidden_init(init_data.float())
        prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
        prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)


        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50
        
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
                yi_ode = ode_sol[:, :, -1, :]

            xi = data[:,i,:].unsqueeze(0)
            
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            prev_y, prev_std = yi, yi_std           
            if run_backwards:
                prev_t, t_i = time_steps[i],  time_steps[i-1]
            elif i < (len(time_steps)-1):
                prev_t,t_i =  time_steps[i], time_steps[i+1]


        return yi, yi_std


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents, 
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.input_dim = input_dim
        self.ode_method = method
        self.latents = latents      
        self.device = device
        self.ode_func = ode_func
        self.odesolver = odeint

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards = False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]
        pred_y,_ = self.odesolver(self.ode_func, first_point, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y

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
class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, 
        update_gate = None,
        reset_gate = None,
        new_state_net = None,
        n_units = 100,
        device = torch.device("cpu")):
        super(GRU_unit, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        if update_gate is None:
            self.update_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim //2, n_units),
               nn.ReLU(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
        else: 
            self.update_gate  = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim//2, n_units),
               nn.ReLU(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim//2, n_units),
               nn.ReLU(),
               nn.Linear(n_units, latent_dim * 2))
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

        if masked_update:


            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()
            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std
        return new_y, new_y_std
