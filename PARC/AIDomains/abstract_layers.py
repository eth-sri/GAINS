import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from collections import Counter
from PARC.AIDomains.zonotope import HybridZonotope
from functools import reduce
from PARC.AIDomains.ai_util import AbstractElement
from typing import Optional, List, Tuple, Union
import PARC.AIDomains.concrete_layers_m as concrete_layers
import pdb


def add_edge(dictionary,key1,key2,value):
    if key1 in dictionary.keys():
        if key2 in dictionary[key1].keys():
            dictionary[key1][key2].append(value) 
        else:
            dictionary[key1][key2] = [value]
    else:
        dictionary[key1] = {key2:[value]}
    return dictionary

def add_vertex(dictionary,key1,value,value2 = None):
    if value2 == None:
        if key1 in dictionary.keys():
            dictionary[key1].append(value) 
        else:
            dictionary[key1] = [value]
    else:
        if key1 in dictionary.keys():
            dictionary[key1][0].append(value) 
            dictionary[key1][1].append(value2) 
        else:
            dictionary[key1] = [[value],[value2]]

    return dictionary

class AbstractModule(nn.Module):
    def __init__(self, save_bounds=True):
        super(AbstractModule, self).__init__()
        self.save_bounds = save_bounds
        self.bounds = None
        self.dim = None

    def update_bounds(self, bounds, detach=True, leave_dim = False):
        lb, ub = bounds

        if detach:
            lb, ub = lb.detach(), ub.detach()

        if not leave_dim:
            try :
                lb = lb.view(-1, *self.dim)
                ub = ub.view(-1, *self.dim)
            except:
                lb = lb.view(-1, *self.output_dim)
                ub = ub.view(-1, *self.output_dim)

        self.bounds = (lb, ub)

    def reset_bounds(self):
        self.bounds = None

    def reset_dim(self):
        self.dim = None
    def get_lambda(self, filtered=False):
        return None

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])
        return self.forward(x)


class Sequential(AbstractModule):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = None

    @classmethod
    def from_concrete_network(
            cls,
            network: nn.Sequential,
            input_dim: Tuple[int, ...],
    ) -> "Sequential":
        abstract_layers: List[AbstractModule] = []
        for i, layer in enumerate(network.children()):
            if i == 0:
                current_layer_input_dim = input_dim
            else:
                current_layer_input_dim = abstract_layers[-1].output_dim
            if isinstance(layer, nn.Sequential):
                abstract_layers.append(Sequential.from_concrete_network(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Linear):
                abstract_layers.append(Linear.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, concrete_layers.Bias):
                abstract_layers.append(Bias.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, concrete_layers.Normalization) or "{0}".format(type(layer)) == "<class 'model.Normalization'>":
                abstract_layers.append(Normalization.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, concrete_layers.DeNormalization):
                abstract_layers.append(DeNormalization.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.ReLU):
                abstract_layers.append(ReLU.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Sigmoid):
                abstract_layers.append(Sigmoid.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Conv2d):
                abstract_layers.append(Conv2d.from_concrete_layer(layer, current_layer_input_dim))
            elif "{0}".format(type(layer)) == "<class 'model.ConcatConv2d'>":
                abstract_layers.append(ConcatConv.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Flatten) or "{0}".format(type(layer)) == "<class 'model.Flatten'>":
                abstract_layers.append(Flatten.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.AvgPool2d):
                abstract_layers.append(AvgPool2d.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.MaxPool2d):
                abstract_layers.append(MaxPool2d.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.Identity):
                abstract_layers.append(Identity(current_layer_input_dim))
            elif isinstance(layer, nn.BatchNorm2d):
                abstract_layers.append(BatchNorm2d.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.GroupNorm):
                abstract_layers.append(GroupNorm.from_concrete_layer(layer, current_layer_input_dim))
            elif "{0}".format(type(layer)) == "<class 'model.ODEBlock'>":
                abstract_layers.append(ODEBlock_A.from_concrete_layer(layer, current_layer_input_dim))
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                abstract_layers.append(GlobalAvgPool2d.from_concrete_layer(layer, current_layer_input_dim))
            elif "{0}".format(type(layer)) == "<class 'model.BatchNorm'>":
                abstract_layers.append(BatchNorm.from_concrete_layer(layer, current_layer_input_dim))
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
        return Sequential(*abstract_layers)

    def forward_between(self, i_from, i_to, x):
        for layer in self.layers[i_from:i_to]:
            x = layer(x)
        return x

    def forward_until(self, i, x):
        return self.forward_between(0, i+1, x)

    def forward_from(self, i, x):
        return self.forward_between(i+1, len(self.layers), x)

    #overload to deal with odesolver
    def forward(self, x,t = None):
        if t == None:
            return self.forward_from(-1, x)
        else:
            out = t.clone()
            time = x
            for i in range(len(self)):
                if self[i].requires_time:
                    out = self[i](time,out)
                else:
                    out = self[i](out)
            return out
        

    def reset_bounds(self, i_from=0, i_to=-1):
        self.bounds = None
        i_to = i_to+1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            layer.reset_bounds()

    def reset_dim(self, i_from=0, i_to=-1):
        i_to = i_to+1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            layer.reset_dim()

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])
        for layer in self.layers:
            x = layer.set_dim(x)
        return x

    def get_lambda(self, filtered=False):
        lambdas = []
        for layer in self.layers:
            lambda_layer = layer.get_lambda(filtered=filtered)
            if lambda_layer is not None:
                lambdas += lambda_layer
        return lambdas


    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]


class Conv2d(nn.Conv2d, AbstractModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dim=None):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.dim = dim

    def forward(self, x):
        if isinstance(x, AbstractElement):
            return x.conv2d(self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super(Conv2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Conv2d, input_dim: Tuple[int, ...]
    ) -> "Conv2d":
        abstract_layer = cls(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
        )
        abstract_layer.weight.data = layer.weight.data
        if layer.bias is not None:
            abstract_layer.bias.data = layer.bias.data

        abstract_layer.output_dim = abstract_layer.getShapeConv(input_dim)
        abstract_layer.dim = input_dim
        abstract_layer.needs_bounds = False

        return abstract_layer

    def getShapeConv(self, input_dim):
        inChan, inH, inW = input_dim
        kH, kW = self.kernel_size

        outH = 1 + int((2 * self.padding[0] + inH - kH) / self.stride[0])
        outW = 1 + int((2 * self.padding[1] + inW - kW) / self.stride[1])
        return (self.out_channels, outH, outW)

class Upsample(nn.Upsample, AbstractModule):
    def __init__(self, size, mode="nearest", align_corners=False, consolidate_errors=False):
        align_corners = None if mode in ["nearest", "area"] else align_corners
        super(Upsample, self).__init__(size=size, mode=mode, align_corners=align_corners)
        self.consolidate_errors = consolidate_errors

    def forward(self, x):
        if isinstance(x, AbstractElement):
            x = x.upsample(size=self.size, mode=self.mode, align_corners=self.align_corners, consolidate_errors=self.consolidate_errors)
        else:
            return super(Upsample, self).forward(x)
        return x


class ReLU(nn.ReLU, AbstractModule):
    def __init__(self, dim: Optional[Tuple]=None) -> None:
        super(ReLU, self).__init__()
        self.deepz_lambda = None if dim is None else nn.Parameter(0.5* torch.ones(dim, dtype=torch.float))
    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement,Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.relu(self.deepz_lambda, self.bounds)
            if deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        else:
            return super(ReLU, self).forward(x)

    def get_lambda(self, filtered=False):
        lb,ub = self.bounds
        if filtered and (((lb<0) * (ub>0)).sum()> 0):
            return [self.deepz_lambda]
        else:
            return None

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.ReLU, input_dim: Tuple[int, ...]
    ) -> "ReLU":
        abstract_layer = cls(input_dim)
        abstract_layer.output_dim = input_dim
        abstract_layer.needs_bounds = True
        abstract_layer.requires_time = False
        return abstract_layer

class Sigmoid(nn.Sigmoid, AbstractModule):
    def __init__(self, dim: Optional[Tuple]=None) -> None:
        super(Sigmoid, self).__init__()

    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement,Tensor]:
        if isinstance(x, AbstractElement):
            out = x.sigmoid(None,None)
            return out
        else:
            return super(Sigmoid, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Sigmoid, input_dim: Tuple[int, ...]
    ) -> "Sigmoid":
        abstract_layer = cls(input_dim)
        abstract_layer.output_dim = input_dim
        abstract_layer.needs_bounds = True
        abstract_layer.requires_time = False
        return abstract_layer

class MaxPool2d(nn.MaxPool2d, AbstractModule):
    def __init__(self, k:int, s:Optional[int]=None, p:Optional[int]=0, d:Optional[int]=1,dim: Optional[Tuple]=None) -> None:
        super(MaxPool2d, self).__init__(kernel_size=k, stride=s, padding=p, dilation=d)
        self.dim=dim

    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement,Tensor]:
        if isinstance(x, AbstractElement):
            assert self.padding == 0
            assert self.dilation == 1
            out = x.max_pool2d(self.kernel_size, self.stride)
            return out
        else:
            return super(MaxPool2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.MaxPool2d, input_dim: Tuple[int, ...]
    ) -> "ReLU":
        abstract_layer = cls(layer.kernel_size, layer.stride, layer.padding, layer.dilation, input_dim)
        abstract_layer.output_dim = abstract_layer.getShapeConv(input_dim)

        return abstract_layer

    def getShapeConv(self, input_dim):
        inChan, inH, inW = input_dim
        kH = kW = self.kernel_size

        outH = 1 + int((2 * self.padding + inH - kH) / self.stride)
        outW = 1 + int((2 * self.padding + inW - kW) / self.stride)
        return (inChan, outH, outW)

class Identity(nn.Identity, AbstractModule):
    def __init__(self, input_dim: Tuple[int, ...]) -> None:
        super(Identity, self).__init__()
        self.output_dim = input_dim

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x

class Log(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None):
        super(Log, self).__init__()
        self.deepz_lambda = nn.Parameter(-torch.ones(dim, dtype=torch.float))

    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.log(self.deepz_lambda, self.bounds)
            if self.deepz_lambda is not None and deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return x.log()

class Exp(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None):
        super(Exp, self).__init__()
        self.deepz_lambda = nn.Parameter(-torch.ones(dim, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.exp(self.deepz_lambda, self.bounds)
            if self.deepz_lambda is not None and deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return x.exp()


class Inv(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None):
        super(Inv, self).__init__()
        self.deepz_lambda = nn.Parameter(-torch.ones(dim, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.inv(self.deepz_lambda, self.bounds)
            if self.deepz_lambda is not None and deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return 1./x


class LogSumExp(AbstractModule):
    def __init__(self, dim=None):
        super(LogSumExp, self).__init__()
        self.dims = dim
        self.exp = Exp(dim)
        self.log = Log(1)
        self.c = None # for MILP verificiation

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dim)

    def reset_bounds(self):
        self.log.bounds = None
        self.bounds = None
        self.exp.bounds = None
        self.c = None

    def set_bounds(self, x):
        self.dim = torch.tensor(x.shape[1:])
        exp_sum = self.exp.set_dim(x).sum(dim=1).unsqueeze(dim=1)
        log_sum = self.log.set_dim(exp_sum)
        return log_sum

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            max_head = x.max_center().detach()
            self.c = max_head
            x_temp = x - max_head
            if self.save_bounds:
                self.exp.update_bounds(x_temp.concretize())
            exp_sum = self.exp(x_temp).sum(dim=1)
            if self.save_bounds:
                self.log.update_bounds(exp_sum.concretize())
            log_sum = self.log(exp_sum)
            return log_sum+max_head
        max_head = x.max(dim=1)[0].unsqueeze(1)
        self.c = max_head
        x_tmp = x-max_head
        exp_sum = x_tmp.exp().sum(dim=1).unsqueeze(dim=1)
        log_sum = exp_sum.log() + max_head
        return log_sum


class Entropy(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None, low_mem:bool=False, neg:bool=False):
        super(Entropy, self).__init__()
        self.exp = Exp(dim)
        self.log_sum_exp = LogSumExp(dim)
        self.low_mem = low_mem
        self.out_features = 1
        self.neg = neg

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def reset_bounds(self):
        self.log_sum_exp.reset_bounds()
        self.bounds = None
        self.exp.bounds = None

    def set_bounds(self, x):
        self.dim = torch.tensor(x.shape[1:])
        log_sum = self.log_sum_exp.set_dim(x)
        softmax = self.exp.set_dim(x - log_sum)
        prob_weighted_act = (softmax * x).sum(dim=1).unsqueeze(dim=1)
        entropy = log_sum - prob_weighted_act
        return entropy

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, HybridZonotope):
            log_sum = self.log_sum_exp(x)
            x_temp = x.add(-log_sum, shared_errors=None if x.errors is None else x.errors.size(0))
            if self.save_bounds:
                self.exp.update_bounds(x_temp.concretize())
            softmax = self.exp(x_temp)
            prob_weighted_act = softmax.prod(x, None if x.errors is None else x.errors.size(0), low_mem=self.low_mem).sum(dim=1)
            entropy = log_sum.add(-prob_weighted_act, shared_errors=None if log_sum.errors is None else log_sum.errors.size(0))
            return entropy * torch.FloatTensor([1-2*self.neg]).to(entropy.head.device)
        log_sum = self.log_sum_exp(x)
        softmax = (x-log_sum).exp()
        prob_weighted_act = (softmax*x).sum(dim=1).unsqueeze(dim=1)
        entropy = log_sum - prob_weighted_act
        return entropy * (1-2*self.neg)


class Flatten(AbstractModule):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.flatten()
        else:
            return x.flatten(1)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Flatten, input_dim: Tuple[int, ...]
    ) -> "Flatten":
        abstract_layer = cls()
        abstract_layer.output_dim = input_dim
        abstract_layer.needs_bounds = False
        return abstract_layer


class Linear(nn.Linear,AbstractModule):
    def __init__(self, in_features:int, out_features:int, bias:bool=True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.linear(self.weight, self.bias)
        return super(Linear, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Linear, input_dim: Tuple[int, ...]
    ) -> "Linear":
        abstract_layer = cls(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
        )
        abstract_layer.weight.data = layer.weight.data
        if layer.bias is not None:
            abstract_layer.bias.data = layer.bias.data
        abstract_layer.output_dim = (layer.weight.shape[0],)
        abstract_layer.needs_bounds = False
        abstract_layer.requires_time = False

        return abstract_layer

class ConcatConv(concrete_layers.ConcatConv2d, AbstractModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dim=None):
        super(ConcatConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.dim = dim

    def forward(self,t, x):
        if isinstance(x, AbstractElement):
            if isinstance(t,AbstractElement):
                temp = x.cat([t,x])
            else:
                temp = x.cat_tensor(t)
            return temp.conv2d(self._layer.weight, self._layer.bias, self._layer.stride, self._layer.padding, self._layer.dilation, self._layer.groups)
        else:
            return super(ConcatConv, self).forward(t,x)

    def getShapeConv(self, input_dim):
        inChan, inH, inW = input_dim
        inChan += 1
        kH, kW = self._layer.kernel_size

        outH = 1 + int((2 * self._layer.padding[0] + inH - kH) / self._layer.stride[0])
        outW = 1 + int((2 * self._layer.padding[1] + inW - kW) / self._layer.stride[1])
        return (self._layer.out_channels, outH, outW)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Conv2d, input_dim: Tuple[int, ...]
    ) -> "Conv2d":
        abstract_layer = cls(
            layer._layer.in_channels -1,
            layer._layer.out_channels,
            layer._layer.kernel_size,
            layer._layer.stride,
            layer._layer.padding,
            layer._layer.dilation,
            layer._layer.groups,
            layer._layer.bias is not None,
        )
        abstract_layer._layer.weight.data = layer._layer.weight.data
        if layer._layer.bias is not None:
            abstract_layer._layer.bias.data = layer._layer.bias.data

        abstract_layer._layer.output_dim = abstract_layer.getShapeConv(input_dim)
        abstract_layer.output_dim = abstract_layer._layer.output_dim 
        abstract_layer._layer.dim = input_dim
        abstract_layer.needs_bounds = False
        abstract_layer.requires_time = True

        return abstract_layer

class GroupNorm(nn.GroupNorm, AbstractModule):
    def __init__(self, num_groups:int, num_channels:int, eps:float=1e-5, affine:bool=True):
        super(GroupNorm, self).__init__(num_groups,num_channels,eps, affine=affine)

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.group_norm(self.num_groups,self.weight,self.bias,self.eps)
        else:
            print("not implemented yet")
            print(aha)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.GroupNorm, input_dim: Tuple[int, ...]
    ) -> "GroupNorm":
        abstract_layer = cls(
            layer.num_groups,
            layer.num_channels,
            layer.eps,
            layer.affine,
        )
        if layer.affine:
            abstract_layer.weight.data = layer.weight.data
            abstract_layer.bias.data = layer.bias.data

        abstract_layer.training = False
        abstract_layer.output_dim = input_dim
        abstract_layer.needs_bounds = True

        return abstract_layer




class _BatchNorm(nn.modules.batchnorm._BatchNorm, AbstractModule):
    def __init__(self, out_features:int, dimensions:int, affine:bool=False):
        super(_BatchNorm, self).__init__(out_features, affine=affine)
        self.current_mean = None
        self.current_var = None
        self.affine = affine
        if not self.affine:
            self.weight = 1
            self.bias = 0
        if dimensions == 1:
            self.mean_dim = [0]
            self.view_dim = (1, -1)
        if dimensions == 2:
            self.mean_dim = [0, 2, 3]
            self.view_dim = (1, -1, 1, 1)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.batch_norm(self)
        if self.training:
            momentum = 1 if self.momentum is None else self.momentum
            self.current_mean = x.mean(dim=self.mean_dim).detach()
            self.current_var = x.var(unbiased=False, dim=self.mean_dim).detach()
            if self.track_running_stats:
                if self.running_mean is not None and self.running_var is not None:
                    self.running_mean = self.running_mean * (1 - momentum) + self.current_mean * momentum
                    self.running_var = self.running_var * (1 - momentum) + self.current_var * momentum
                else:
                    self.running_mean = self.current_mean
                    self.running_var = self.current_var
        else:
            self.current_mean = self.running_mean
            self.current_var = self.running_var
        c = (self.weight / torch.sqrt(self.current_var + self.eps))
        b = (-self.current_mean * c + self.bias)
        return x*c.view(self.view_dim)+b.view(self.view_dim)

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d], input_dim: Tuple[int, ...]
    ) -> "_BatchNorm":
        abstract_layer = cls(
            layer.num_features,
            layer.affine,
        )
        abstract_layer.running_var.data = layer.running_var.data
        abstract_layer.running_mean.data = layer.running_mean.data
        if layer.affine:
            abstract_layer.weight.data = layer.weight.data
            abstract_layer.bias.data = layer.bias.data

        abstract_layer.track_running_stats = layer.track_running_stats
        abstract_layer.training = False
        abstract_layer.output_dim = input_dim
        abstract_layer.needs_bounds = False

        return abstract_layer


class BatchNorm1d(_BatchNorm):
    def __init__(self, out_features:int, affine:bool=False):
        super(BatchNorm1d, self).__init__(out_features, 1, affine)


class BatchNorm2d(_BatchNorm):
    def __init__(self, out_features:int, affine:bool=False):
        super(BatchNorm2d, self).__init__(out_features, 2, affine)


class AvgPool2d(nn.AvgPool2d, AbstractModule):
    def __init__(self, kernel_size:int, stride:Optional[int]=None, padding:int=0):
        super(AvgPool2d, self).__init__(kernel_size, stride, padding)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            assert self.padding == 0
            return x.avg_pool2d(self.kernel_size, self.stride)
        return super(AvgPool2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[nn.AvgPool2d], input_dim: Tuple[int, ...]
    ) -> "AvgPool2d":
        abstract_layer = cls(
            layer.kernel_size,
            layer.stride,
            layer.padding,
        )
        abstract_layer.output_dim = input_dim

        return abstract_layer


class GlobalAvgPool2d(nn.AdaptiveAvgPool2d, AbstractModule):
    def __init__(self,output_size):
        super(GlobalAvgPool2d, self).__init__(output_size)

        self.output_size = output_size

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.global_avg_pool2d(self.output_size)
        return super(GlobalAvgPool2d, self).forward(x)
    @classmethod
    def from_concrete_layer(
        cls, layer: Union[nn.AdaptiveAvgPool2d], input_dim: Tuple[int, ...]
    ) -> "GlobalAvgPool2d":
        abstract_layer = cls(
            layer.output_size
        )
        abstract_layer.output_size = layer.output_size
        abstract_layer.output_dim = (input_dim[0],)
        abstract_layer.input_dim = input_dim
        abstract_layer.needs_bounds = False
        return abstract_layer

class Bias(AbstractModule):
    def __init__(self, bias=0, fixed=False):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(bias*torch.ones(1))
        self.bias.requires_grad_(not fixed)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x + self.bias

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[concrete_layers.Bias], input_dim: Tuple[int, ...]
    ) -> "Bias":
        abstract_layer = cls(layer.bias)
        abstract_layer.output_dim = input_dim
        return abstract_layer


class Scale(AbstractModule):
    def __init__(self, scale=1, fixed=False):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(scale*torch.ones(1))
        self.scale.requires_grad_(not fixed)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x * self.scale


class Normalization(AbstractModule):
    def __init__(self, mean, sigma):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.sigma = nn.Parameter(sigma, requires_grad=False)
        self.mean.requires_grad_(False)
        self.sigma.requires_grad_(False)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        target_shape = [1,-1] + (x.dim()-2) * [1]
        return (x - self.mean.view(target_shape)) *(1/ self.sigma.view(target_shape))

    @classmethod
    def from_concrete_layer(
        cls, layer: concrete_layers.Normalization, input_dim: Tuple[int, ...]
    ) -> "Normalization":
        abstract_layer = cls(layer.mean, layer.sigma)
        abstract_layer.output_dim = input_dim
        abstract_layer.needs_bounds = False
        return abstract_layer


class DeNormalization(AbstractModule):
    def __init__(self, mean, sigma):
        super(DeNormalization, self).__init__()
        self.mean = nn.Parameter(mean)
        self.sigma = nn.Parameter(sigma)
        self.mean.requires_grad_(False)
        self.sigma.requires_grad_(False)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        target_shape = [1,-1] + (x.dim()-2) * [1]
        return x * self.sigma.view(target_shape) + self.mean.view(target_shape)

    @classmethod
    def from_concrete_layer(
        cls, layer: concrete_layers.DeNormalization, input_dim: Tuple[int, ...]
    ) -> "DeNormalization":
        abstract_layer = cls(layer.mean, layer.std)
        abstract_layer.output_dim = input_dim
        return abstract_layer

class DomainEnforcing(Sequential):
    def __init__(self, input_dim:Union[Tensor, List[int]],
                 mean: Optional[Union[Tensor, List[float], float]] = None,
                 std: Optional[Union[Tensor, List[float], float]] = None) -> "DomainEnforcing":
        layers = []
        if mean is not None or std is not None:
            layers += [DeNormalization(mean, std)]
        layers += [ReLU()]
        layers += [Normalization(torch.tensor([1.]), torch.tensor([-1.]))]
        layers += [ReLU()]
        layers += [DeNormalization(torch.tensor([1.]), torch.tensor([-1.]))]
        if mean is not None or std is not None:
            layers += [Normalization(mean, std)]

        super(DomainEnforcing, self).__init__(*layers)

    @classmethod
    def enforce_domain(cls, x, mean: Optional[Union[Tensor, List[float], float]] = None,
                 std: Optional[Union[Tensor, List[float], float]] = None):
        input_dim = torch.tensor(x.shape).numpy().tolist()
        layer = cls(input_dim, mean, std).to(device=x.device)
        return layer(x)

class ResBlock(AbstractModule):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None, mode="standard"):
        super(ResBlock, self).__init__()

        self.residual = self.get_residual_layers(mode, in_planes, planes, stride, dim)
        self.downsample = downsample
        self.relu_final = ReLU((planes, dim//stride, dim//stride)) if mode in ["standard"] else None

    def forward(self, x):
        identity = x.clone()
        out = self.residual(x)

        if self.downsample is not None:
            if not isinstance(self.downsample, Sequential):
                if isinstance(x, AbstractElement) and self.downsample.save_bounds:
                    self.downsample.update_bounds(x.concretize(), detach=True)
            identity = self.downsample(x)

        if isinstance(out, HybridZonotope):
            out = out.add(identity, shared_errors=None if identity.errors is None else identity.errors.size(0))
        elif isinstance(out, AbstractElement):
            out = out.add(identity)
        else:
            out += identity
        if self.relu_final is not None:
            if isinstance(out, AbstractElement) and self.relu_final.save_bounds:
                self.relu_final.update_bounds(out.concretize(), detach=True)
            out = self.relu_final(out)
        return out

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])

        identity = x.clone()
        out = self.residual.set_dim(x)
        if self.downsample is not None:
            out += self.downsample.set_dim(identity)
        else:
            out += identity
        if self.relu_final is not None:
            out = self.relu_final.set_dim(out)
        return out

    def reset_bounds(self):
        self.bounds = None
        if self.downsample is not None:
            self.downsample.reset_bounds()

        if self.relu_final is not None:
            self.relu_final.reset_bounds()

        self.residual.reset_bounds()

    def get_residual_layers(self, mode, in_planes, out_planes, stride, dim):
        if mode == "standard":
            residual = Sequential(
                Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim // stride, dim // stride)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm2d(out_planes),
            )
        elif mode == "wide":
            residual = Sequential(
                BatchNorm2d(in_planes),
                ReLU((in_planes, dim, dim)),
                Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim, dim)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True),
            )
        elif mode == "fixup":
            residual = Sequential(
                Bias(),
                Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                Bias(),
                ReLU((out_planes, dim // stride, dim // stride)),
                Bias(),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                Scale(),
                Bias(),
            )
        elif mode =="test":
            residual = Sequential(
                # Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim // stride, dim // stride)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            raise RuntimeError(f"Unknown layer mode {mode:%s}")

        return residual


class BasicBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="standard")


class TestBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(TestBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="test")


class WideBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(WideBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="wide")


class FixupBasicBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="fixup")

        if downsample is not None:
            self.downsample = Sequential(
                self.residual.layers[0],
                downsample)


def add_bounds(lidx, zono, bounds=None, layer=None):
    lb_new, ub_new = zono.concretize()
    if layer is not None:
        if layer.bounds is not None:
            lb_old, ub_old = layer.bounds
            lb_new, ub_new = torch.max(lb_old, lb_new).detach(), torch.min(ub_old, ub_new).detach()
        layer.bounds = (lb_new, ub_new)
    if bounds is not None:
        bounds[lidx] = (lb_new, ub_new)
        return bounds

class ODEBlock_A(concrete_layers.ODEBlock, AbstractModule):
    def __init__(self, met:str, integration_time:float,device): 
        super(ODEBlock_A, self).__init__(met,integration_time,device)

        self.method = met
        self.integration_time = integration_time
        self.atol = 0
        self.rtol = 0.0001
        self.device = device

        self.deepz_lambda_dict = {}
        self.use_lambda= False

        self.provable_training_mode = 0

        if self.method == "rk4_10":
            self.steps = 10
            self.dt = integration_time / 10
        if self.method == "rk4_20":
            self.met = "rk4"
            self.steps = 20
            self.dt = integration_time / 20
        elif self.method == "rk4_0.1":
            self.dt = 0.1
            self.steps = integration_time / 0.1
        elif self.method == "euler_2":
            self.met = "euler"
            self.dt = integration_time / 2
            self.steps = 2
        elif self.method == "mp_2":
            self.dt = integration_time / 2
            self.steps = 2
        elif self.method == "rk4_2":
            self.dt = integration_time / 2
            self.steps = 2
        elif self.method == "rk4_1":
            self.alpha = torch.tensor([ 1 / 3, 2 / 3, 1.], dtype=torch.float64)
            self.beta=[
                    torch.tensor([1 / 3], dtype=torch.float64),
                    torch.tensor([-1 / 3, 1.], dtype=torch.float64),
                    torch.tensor([1., -1., 1.], dtype=torch.float64),
                ]
            self.c_sol =torch.tensor([1/8, 3/8, 3 / 8, 1 / 8], dtype=torch.float64)
            self.dt = integration_time 
            self.steps = 1
        elif self.method == "euler":
            self.dt = integration_time 
            self.steps = 1
        elif self.method == "dopri5_0.005_1.5a":
            self.adaptive_mult_factor = 1.5
            self.atol = 0.005
            self.tau = 0.15
            self.eta = 0.02
        elif self.method in ["dopri5_0.005_2a","dopri5_0.01_2a","dopri5_0.001_2a","dopri5_0.0001_2a"]:
            # dont need k7 stuff here for deeppoly part
            self.alpha=torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.], dtype=torch.float64)
            self.beta=[
                    torch.tensor([1 / 5], dtype=torch.float64),
                    torch.tensor([3 / 40, 9 / 40], dtype=torch.float64),
                    torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=torch.float64),
                    torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=torch.float64),
                    torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=torch.float64)
                ]
            self.c_sol =torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=torch.float64)
            self.adaptive_mult_factor = 2
            self.met = "dopri5"
            if self.method == "dopri5_0.01_2a":
                self.atol = 0.01
            elif self.method == "dopri5_0.001_2a":
                self.atol = 0.001
            elif self.method == "dopri5_0.005_2a":
                self.atol = 0.005
            else:
                self.atol = 0.0001
            self.rtol = 0
            self.tau = 2 ** (-5)
            self.eta = 0.02
            self.mode = "training"

            self.prob_updates = None
            self.prob_start = 10
            self.prob_end = 80
            self.prob_mid = (self.prob_end + self.prob_start) / 2
            self.prob_len = self.prob_end - self.prob_start
            self.q0,self.p0 = torch.tensor(0.15).to(self.device),torch.tensor(0.15).to(self.device)
            self.qend,self.pend = torch.tensor(1/3).to(self.device),torch.tensor(1/3).to(self.device)
            self.prob_q_a,self.prob_q_b = (self.qend - self.q0)/2, (self.qend + self.q0)/2
            self.prob_p_a,self.prob_p_b = (self.pend - self.p0)/2, (self.pend + self.p0)/2



            self.update_probabilities()

        elif self.method in ["bosh3_0.005_2a"]:
            # dont need k7 stuff here for deeppoly part
            self.alpha=torch.tensor([1 / 2, 3 / 4, 1.], dtype=torch.float64)
            self.beta=[
                    torch.tensor([1 / 2], dtype=torch.float64),
                    torch.tensor([0., 3 / 4], dtype=torch.float64),
                    torch.tensor([2 / 9, 1 / 3, 4 / 9], dtype=torch.float64),
                ]
            self.c_sol =torch.tensor([2 / 9, 1/3, 4 / 9, 0], dtype=torch.float64)
            self.adaptive_mult_factor = 2
            self.met = "bosh3"

            self.atol = 0.005
            self.rtol = 0
            self.tau = 2 ** (-3)
            self.eta = 0.02
            self.mode = "training"
            self.prob_updates = None
            self.prob_start = 10
            self.prob_end = 80
            self.prob_mid = (self.prob_end + self.prob_start) / 2
            self.prob_len = self.prob_end - self.prob_start
            self.q0,self.p0 = torch.tensor(0.15).to(self.device),torch.tensor(0.15).to(self.device)
            self.qend,self.pend = torch.tensor(1/3).to(self.device),torch.tensor(1/3).to(self.device)

            self.prob_q_a,self.prob_q_b = (self.qend - self.q0)/2, (self.qend + self.q0)/2
            self.prob_p_a,self.prob_p_b = (self.pend - self.p0)/2, (self.pend + self.p0)/2

            self.update_probabilities()

        else:
            print("Not implemented method:",self.method)
            exit()
        self.allowed_trajectories = 1
        self.T = 2.5
        self.shift = 2

    def get_lambda(self, filtered=False):
        lambdas = []
        for key in self.deepz_lambda_dict.keys():
            lambdas += [self.deepz_lambda_dict[key]]
        if len(lambdas)> 0:
            return lambdas    
        return None

    def update_qs(self,q_start,q_end):
        self.q0,self.p0 = torch.tensor(q_start).to(self.device),torch.tensor(q_start).to(self.device)
        self.qend,self.pend = torch.tensor(q_end).to(self.device),torch.tensor(q_end).to(self.device)
        self.prob_q_a,self.prob_q_b = (self.qend - self.q0)/2, (self.qend + self.q0)/2
        self.prob_p_a,self.prob_p_b = (self.pend - self.p0)/2, (self.pend + self.p0)/2

        if self.prob_updates < self.prob_start:
            self.q = self.q0
            self.p = self.p0
        elif self.prob_updates >= self.prob_end:
            self.q = self.qend
            self.p = self.pend
        else:
            self.q = np.sin(np.pi* (self.prob_updates -self.prob_mid) / self.prob_len) * self.prob_q_a + self.prob_q_b
            self.p = np.sin(np.pi* (self.prob_updates -self.prob_mid) / self.prob_len) * self.prob_p_a + self.prob_p_b

    def update_probabilities(self):
        if self.prob_updates == None:
            self.prob_updates = 0 
        else:
            self.prob_updates += 1
        if self.prob_updates < self.prob_start:
            self.q = self.q0
            self.p = self.p0
        elif self.prob_updates >= self.prob_end:
            self.q = self.qend
            self.p = self.pend
        else:
            self.q = np.sin(np.pi* (self.prob_updates -self.prob_mid) / self.prob_len) * self.prob_q_a + self.prob_q_b
            self.p = np.sin(np.pi* (self.prob_updates -self.prob_mid) / self.prob_len) * self.prob_p_a + self.prob_p_b

    def forward(self, x:Union[AbstractElement, Tensor],mode2 = False) -> Union[AbstractElement, Tensor]:

        if self.method in ["bosh3_0.005_2a"]:
            bosh3_flag = True
        else:
            bosh3_flag = False
            
        if isinstance(x, AbstractElement):
            if self.method in ["rk4_10","rk4_20","rk4_0.1"]:
                t =  0
                out = x.clone()
                for i in range(self.steps):
                    out = self.rk4_step(out,t,self.dt)
                    t += self.dt
                return out
            elif self.method in ["rk4_1"]:
                t =  0
                graph = {}
                temporary_graph = {(0,-1,False):[x.clone()]}
                key = min(temporary_graph)
                t0,dt,flag = key[0] * self.integration_time, -(key[1]) * self.integration_time, key[2]
                x = temporary_graph.pop(key)[0]

                bounds = {}
                k1,bounds['k1'] = self.odefunc_forward(t0,x,True)#k1 = f(t0,y0)
                temp = x.clone().add(k1*(dt/3))
                k2,bounds['k2'] = self.odefunc_forward(t0 + (dt/3),temp,True) #k2 = f(t0 + 1/3*dt,y0 + 1/3*dt*k1)

                temp = x.clone().add(k2*dt)
                temp = temp.add(k1*(dt*(-1)/3))
                k3,bounds['k3'] = self.odefunc_forward(t0 + (dt*2/3),temp,True) #k3 = f(t0 + 2/3*dt,y0  -1/3*k1*dt + k2*dt)

                temp = x.clone().add(k3*dt)
                temp = temp.add(k2*(dt*(-1)))
                temp = temp.add(k1*(dt))
                k4,bounds['k4'] = self.odefunc_forward(t0 + dt,temp,True) #k4 = f(t0 + 1*dt,y0 +dt*(k1-k2+k3))

                final = x.clone().add(0.125*dt *k1)
                final = final.add(0.125*dt *k4)
                final = final.add(0.375*dt *k2)
                final = final.add(0.375*dt *k3)

                dt_next = key[1]
                t1 = key[0]- key[1]
                graph = add_edge(graph,(t1,dt_next,False),key,[final.clone(),bounds])
                self.graph = graph
                return final
            elif self.mode == "abstract_fixed":
                self.ratio = self.integration_time / self.running_mean_step
                graph = {}
                temporary_x = x.clone()
                temporary_graph = {(0,-1,0,0,False):[x.clone()]}
                for i in range(self.states.shape[0]):
                    key = min(temporary_graph)
                    t0 = key[0] * self.running_mean_step * (1  - key[2]) + self.integration_time * key[2]
                    dt = -(key[1]) * self.running_mean_step  - key[3] * (self.integration_time - key[0] * self.running_mean_step)
                    flag = key[4]
                    y0 = temporary_graph.pop(key)[0]
                    if flag:
                        print("not expected flag in abstract_traj")
                        exit()

                    y1,_,bounds = self.dopri5_step(y0,t0,dt)
                    #step accepted check for clipping
                    if self.trajectory_path[i] > 0:
                        t1 = key[0] - key[1]  
                        dt_next = key[1] * self.trajectory_path[i]
                        t1_c,dt_c = key[2] - key[3], key[3]* self.trajectory_path[i]
                        if dt_next <= t1 - self.ratio:
                            dt_next = 0
                            t1_c,dt_c = 0, -1
                        if dt_c <= t1_c - 1:
                            dt_c = t1_c - 1 
                        
                        if i != (self.states.shape[0]-1):
                            temporary_graph = add_vertex(temporary_graph,(t1,dt_next,t1_c,dt_c,False),y1.clone())
                            graph = add_edge(graph,(t1,dt_next,t1_c,dt_c,False),key,[y1.concretize(),bounds])
                        else:
                            graph = add_edge(graph,(self.ratio.item(),0,0,0,False),key,[y1.concretize(),bounds])

                    else:
                        t1 = key[0]
                        dt_next = key[1] / self.adaptive_mult_factor
                        t1_c,dt_c = key[2], key[3] / self.adaptive_mult_factor

                        if (dt_c) == 0 and dt_next > -(self.ratio)* self.eta:
                            dt_next,flag = key[1], True
                        if (dt_next == 0) and dt_c > -1/2:
                            dt_c,flag  = -1/2, True
                        
                        graph = add_edge(graph,(t1,dt_next,t1_c,dt_c,flag),key,[y0.concretize(),[]])
                        temporary_graph = add_vertex(temporary_graph,(t1,dt_next,t1_c,dt_c,flag),y0.clone())
                self.graph = [graph,temporary_x.concretize()]



            elif self.method in ["euler_2"] and self.mode == "abstract":
                t =  0
                self.dt = self.integration_time /2
                graph = {}
                temporary_x = x.clone()
                temporary_graph = {(0,-1,0,0,False):[x.clone()]}
                while len(temporary_graph)> 0:
                    key = min(temporary_graph)
                    t0,dt,flag = key[0] * self.dt, -(key[1]) * self.dt, key[4]
                    x = temporary_graph.pop(key)[0]

                    bounds = {}

                    k1,bounds['k1'] = self.odefunc_forward(t0,x,True)#k1 = f(t0,y0)

                    final = x.clone().add(dt *k1)

                    dt_next = key[1]
                    t1 = key[0]- key[1]

                    if (t1* self.dt) + 1e-8 <  self.integration_time:
                        graph = add_edge(graph,(t1,dt_next,0,0,False),key,[final.clone(),bounds])
                        temporary_graph = add_vertex(temporary_graph,(t1,dt_next,0,0,False),final.clone())
                    else:
                        graph = add_edge(graph,(2,0,0,0,False),key,[final.concretize(),bounds])
                self.graph = [graph,temporary_x.concretize()]
                return final

            elif self.method in ["euler_2", "euler"]:
                t =  0
                out = x.clone()
                for i in range(self.steps):
                    out = self.euler_step(out,t,self.dt)
                    t += self.dt
                return out
            elif self.method in ["dopri5_0.005_1.5a","dopri5_0.005_2a","dopri5_0.001_2a","dopri5_0.01_2a","dopri5_0.0001_2a","bosh3_0.005_2a"] and self.mode == "abstract":
                #assumes x to be box/zonotope
                tol = x.shape[0]* self.atol
                self.ratio = self.integration_time / self.running_mean_step
                clipped = False
                #due to getting min key entries 1 and 3 are stored negatively
                temporary_graph = {(0,-1,0,-0,False):[x.clone()]}

                mode2 = True
                if mode2:
                    final_lower = None 
                    final_upper = None

                graph = {}

                while len(temporary_graph) > 0:
                    key = min(temporary_graph)
                    t0 = key[0] * self.running_mean_step * (1  - key[2]) + self.integration_time * key[2]
                    dt = -(key[1]) * self.running_mean_step  - key[3] * (self.integration_time - key[0] * self.running_mean_step)
                    flag = key[4]

                    y0 = temporary_graph.pop(key)
                    if len(y0) == 1:
                        y0 = y0[0]
                    else:
                        lb,ub = y0[0].concretize()
                        for j in range(1,len(y0)):
                            temp = y0[j].concretize()

                            lb = torch.minimum(lb,temp[0])
                            ub = torch.maximum(ub,temp[1])
                        y0 = HybridZonotope.construct_from_bounds(min_x= lb, 
                                    max_x= ub, 
                                    dtype = lb.dtype,
                                    domain = "box")

                    #Actual forward pass
                    if bosh3_flag:
                        y1,err,bounds = self.bosh3_step(y0,t0,dt)
                    else:
                        y1,err,bounds = self.dopri5_step(y0,t0,dt,key)

                    if flag:#step size too low issue
                        t1 = key[0] - key[1]
                        dt_next = key[1] 
                        t1_c,dt_c = key[2] - key[3], key[3]

                        if dt_next <= t1 - self.ratio:
                            dt_next = 0
                            t1_c,dt_c = 0, -1
                        #clipp second part of key
                        if dt_c <= t1_c - 1:
                            dt_c = t1_c - 1 

                        if (t1* self.running_mean_step) * (1 - t1_c) + 1e-8 <  self.integration_time * (1 - t1_c):
                            temporary_graph = add_vertex(temporary_graph,(t1,dt_next,t1_c,dt_c,True),y1.clone())
                            graph = add_edge(graph,(t1,dt_next,t1_c,dt_c,True),key,[y1.concretize(),bounds])
                        else:
                            graph = add_edge(graph,(self.ratio.item(),0,0,0,False),key,[y1.concretize(),bounds])
                            if mode2:
                                if final_lower == None:
                                    final_lower,final_upper = y1.concretize()
                                else:
                                    final_lower = torch.minimum(final_lower,y1.concretize()[0])
                                    final_upper = torch.maximum(final_upper,y1.concretize()[1])
                    else:
                        #error_calculations
                        err_lb,err_ub = err.abs(mode= True)
                        err_lb,err_ub = err_lb.sum()/tol,err_ub.sum()/tol

                        if (err_lb <= 1.0 and err_ub > self.tau):
                            #at any time either key[1] or key[3] is equal to 0
                            t1 = key[0] - key[1]
                            dt_next = key[1] 
                            t1_c,dt_c = key[2] - key[3], key[3]
                            if dt_next <= t1 - self.ratio:
                                dt_next = 0
                                t1_c,dt_c = 0, -1
                                clipped = True

                            if dt_c <= t1_c - 1:
                                dt_c = t1_c - 1 
                                clipped = True

                            if (t1* self.running_mean_step) * (1 - t1_c) + 1e-8 <  self.integration_time * (1 - t1_c):
                                graph = add_edge(graph,(t1,dt_next,t1_c,dt_c,False),key,[y1.concretize(),bounds])
                                temporary_graph = add_vertex(temporary_graph,(t1,dt_next,t1_c,dt_c,False),y1.clone())
                            else:
                                clipped = True
                                graph = add_edge(graph,(self.ratio.item(),0,0,0,False),key,[y1.concretize(),bounds])
                                if mode2:
                                    if final_lower == None:
                                        final_lower,final_upper = y1.concretize()
                                    else:
                                        final_lower = torch.minimum(final_lower,y1.concretize()[0])
                                        final_upper = torch.maximum(final_upper,y1.concretize()[1])

                        if err_lb <= self.tau and not clipped:
                            t1 = key[0] - key[1]
                            dt_next = key[1] * self.adaptive_mult_factor
                            t1_c,dt_c = key[2] - key[3], key[3] * self.adaptive_mult_factor

                            if dt_next <= t1 - self.ratio:
                                dt_next = 0
                                t1_c,dt_c = 0, -1

                            if dt_c <= t1_c - 1:
                                dt_c = t1_c - 1 

                            if (t1* self.running_mean_step) * (1 - t1_c) + 1e-8 <  self.integration_time * (1 - t1_c):
                                graph = add_edge(graph,(t1,dt_next,t1_c,dt_c,False),key,[y1.concretize(),bounds])
                                temporary_graph = add_vertex(temporary_graph,(t1,dt_next,t1_c,dt_c,False),y1.clone())
                            else:
                                graph = add_edge(graph,(self.ratio.item(),0,0,0,False),key,[y1.concretize(),bounds])
                                if mode2:
                                    if final_lower == None:
                                        final_lower,final_upper = y1.concretize()
                                    else:
                                        final_lower = torch.minimum(final_lower,y1.concretize()[0])
                                        final_upper = torch.maximum(final_upper,y1.concretize()[1])

                        if err_ub >  1.0:
                            t1 = key[0]
                            dt_next = key[1] / self.adaptive_mult_factor
                            t1_c,dt_c = key[2], key[3] / self.adaptive_mult_factor

                            if (dt_c) == 0 and dt_next > -(self.ratio)* self.eta:
                                dt_next,flag = key[1], True
 
                            if (dt_next == 0) and dt_c > -1/2:
                                dt_c,flag  = -1/2, True
                            graph = add_edge(graph,(t1,dt_next,t1_c,dt_c,flag),key,[y0.concretize(),[]])
                            temporary_graph = add_vertex(temporary_graph,(t1,dt_next,t1_c,dt_c,flag),y0.clone())
                    clipped = False

                self.graph = [graph,x.concretize()]
                if mode2:
                    return HybridZonotope.construct_from_bounds(min_x= final_lower, max_x= final_upper, 
                                dtype = final_lower.dtype,domain = "box")
                
            elif self.method in ["dopri5_0.005_1.5a","dopri5_0.005_2a","dopri5_0.01_2a","dopri5_0.001_2a","dopri5_0.0001_2a","bosh3_0.005_2a"] and self.mode == "training":
                #assumes x to be box
                tol = x.shape[0]* self.atol
                if self.last_step == None:
                    self.last_step = self.running_mean_step

                temporary_graph = [[[], x.clone(), 0*self.last_step, self.last_step,False,0,0]]
                graph = []
                location_list = []
                self.splits = 0
                self.error_penalty = torch.zeros(1).to(self.device)
                while len(temporary_graph) > 0:
                    if len(location_list) == 0:
                        mean = 0
                    else:
                        mean = np.mean(location_list)

                    idx = max(temporary_graph, key=lambda x: np.abs(x[-2] -mean)* 0.5 -len(x[0])- x[-1])
                    key = idx
                    idx = temporary_graph.index(idx)
                    trajectory, y0, t0, dt, flag,location_idx,branches = temporary_graph.pop(idx)

                    while t0 < self.integration_time:
                        delta = 0
                        if bosh3_flag:
                            y1,err,bounds = self.bosh3_step(y0,t0,dt)
                        else:
                            y1,err,bounds = self.dopri5_step(y0,t0,dt,key)

                        if flag:
                            t1 = t0 + dt 
                            dt_next = dt 
                            if self.integration_time - t1 < dt_next:
                                dt_next = self.integration_time - t1
                            trajectory += [1]
                            t0,dt = t1, dt_next
                            y0 = y1

                        else:
                            err_lbb,err_ubb = err.abs(mode= True)
                            err_lb,err_ub = err_lbb.sum()/tol,err_ubb.sum()/tol

                            if err_lb <= self.tau and err_ub > 1: # all three possible
                                case = 3 
                                self.splits += 2
                                actions = [0,1,2]
                                #resolve clipped case, in order to avoid double trajectory
                                if (self.integration_time - t0 - dt < dt) or ((t0 - self.integration_time).abs() < 1e-8):
                                    case = 1
                                    self.splits -= 1
                                    actions = [0,1]

                                ref_action = self.get_reference_action(torch.tensor([t0,dt]).to(self.device))
                                probabilities = self.get_prob(location_idx,ref_action,case)
                                sampled_action = torch.multinomial(probabilities,1)

                            elif err_lb <= self.tau and err_ub <= 1 and err_ub > self.tau: #both accept
                                self.splits += 1
                                actions = [1,2]
                                #resolve clipped case, in order to avoid double trajectory
                                if (self.integration_time - t0 - dt < dt) or ((t0 - self.integration_time).abs() < 1e-8):
                                    self.splits -= 1
                                    actions = [1]
                                    ref_action = 1
                                    sampled_action = 1
                                else:
                                    ref_action = self.get_reference_action(torch.tensor([t0,dt]).to(self.device))
                                    probabilities = self.get_prob(location_idx,ref_action,2)
                                    sampled_action = torch.multinomial(probabilities,1)

                            elif err_lb > self.tau and err_ub > 1 and err_lb <= 1: #reject and accept possible
                                self.splits += 1
                                actions = [0,1]
                                ref_action = self.get_reference_action(torch.tensor([t0,dt]).to(self.device))
                                probabilities = self.get_prob(location_idx,ref_action,1)
                                sampled_action = torch.multinomial(probabilities,1)

                            elif (err_lb <= 1.0 and err_ub > self.tau): # accept
                                actions = [1]
                                ref_action = 1
                                sampled_action = 1
                            elif err_ub <= self.tau: #increase
                                actions = [2]
                                ref_action = 2
                                sampled_action = 2
                            elif err_lb > 1:
                                actions = [0]
                                ref_action = 0
                                sampled_action = 0
                            else:
                                print("Should not have happened")
                                exit()

                            if len(actions) > 1:
                                B = y1.shape[0]
                                err_length = (err_ubb - err_lbb).flatten() * np.exp(-self.T * (branches - self.shift))
                                self.error_penalty += err_length.topk(B*10)[0].mean()

                            #fix first trajectory to be reference trajectory and then allow kappa -1 trajectories
                            if self.provable_training_mode == 3 and len(graph) == 0:
                                sampled_action = ref_action
                            for action in actions:
                                if action == 0:
                                    temp_trajectory = trajectory + [0]
                                    t1_temp,flag_temp = t0,flag
                                    dt_next_temp = dt / self.adaptive_mult_factor
                                    if dt_next_temp < self.integration_time * self.eta:
                                        dt_next_temp,flag_temp = dt, True
                                    y1_temp = y0.clone()

                                elif action == 1:
                                    t1_temp, dt_next_temp, flag_temp = t0 + dt, dt, flag
                                    if self.integration_time - t1_temp < dt_next_temp:
                                        dt_next_temp = self.integration_time - t1_temp
                                    temp_trajectory = trajectory + [1]
                                    y1_temp = y1.clone()
                                else: #action == 2
                                    t1_temp, dt_next_temp, flag_temp = t0 + dt, dt * self.adaptive_mult_factor, flag
                                    if self.integration_time - t1_temp < dt_next_temp:
                                        dt_next_temp = self.integration_time - t1_temp
                                    temp_trajectory = trajectory + [2]
                                    y1_temp = y1.clone()

                                location_idx_temp = location_idx + action - ref_action

                                if action != sampled_action:
                                    #don't add to checkpoints if eitherway probably not selected in order to avoid out of memory error
                                    if len(temporary_graph)> 45 or branches  >= 6:
                                        delta += 0
                                    else:
                                        delta += 1
                                        temporary_graph.append([temp_trajectory, y1_temp, t1_temp, dt_next_temp,flag_temp,location_idx_temp,branches + 1])
                                else:
                                    true_trajectory, y1_true, t1_true, dt_next_true,flag_true,location_idx_true = temp_trajectory, y1_temp, t1_temp, dt_next_temp,flag_temp,location_idx_temp

                            branches += delta
                            trajectory, y0, t0, dt, flag, location_idx = true_trajectory, y1_true, t1_true, dt_next_true,flag_true,location_idx_true
                    location_list.append(location_idx) 
                    graph.append(y0)
                    if len(graph) == self.allowed_trajectories:
                       break

                if self.provable_training_mode in [0,3]:
                    out = HybridZonotope.cat(graph)
                elif self.provable_training_mode == 1:
                    #take average
                    number = len(graph)
                    inv = 1/number
                    out = graph[0]* inv
                    for i in range(1,number):
                        out = out.add(graph[i]*inv)
                elif self.provable_training_mode == 2:
                    #take maximum/minimum bound
                    lb,ub = graph[0].concretize()
                    for i in range(1,len(graph)):
                        lb_temp,ub_temp = graph[i].concretize()

                        lb = torch.minimum(lb,lb_temp)
                        ub = torch.maximum(ub,ub_temp)
                    out = HybridZonotope.construct_from_bounds(min_x= lb, max_x= ub, 
                                dtype = ub.dtype,domain = "box")


                self.trajectory_path = None
                self.last_step = None
                self.states = None
                return out 
            else:
                print("not implemented yet")
                exit()
                
        else:
            if self.method in ["rk4_20","euler_2"]:
                out,self.liste = self.odesolver(self.odefunc, x, torch.tensor([0,self.integration_time]).to(x.device).float(), method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.dt) 
                return out[1]
            if self.training:
                out,self.liste = self.odesolver(self.odefunc, x, torch.tensor([0,self.integration_time]).to(x.device).float(), method = self.met, rtol= self.rtol,atol= self.atol, step_size= self.step_size, adaptive_step_factor = self.adaptive_mult_factor, running_mean_step = None) 
                self.running_mean_step.data = (1-0.1)* self.running_mean_step.data + 0.1 * self.liste[-2]
                self.last_step = self.liste[-2]
            else:
                out,self.liste = self.odesolver(self.odefunc, x, torch.tensor([0,self.integration_time]).to(x.device).float(), method = self.met,rtol= self.rtol,atol= self.atol,step_size= self.step_size,adaptive_step_factor = self.adaptive_mult_factor,running_mean_step = self.running_mean_step.data)
            self.states = self.liste[-3]
            self.trajectory_path = self.liste[-4] 
            self.liste = self.liste[0:-4]
            return out[1]

    def get_reference_action(self,x):
        abc = (self.states - x.view(1,2)).abs().sum(1)
        return self.trajectory_path[abc.argmin()]

    def get_prob(self,location, action, case):
        if location == 0:
            if action == 0:
                probabilities = torch.tensor([1 - self.p - self.q, self.p, self.q]).to(self.device).type(self.dtype)
            elif action == 1:
                probabilities = torch.tensor([(self.p+self.q)/2,1 - self.p - self.q, (self.p+self.q)/2]).to(self.device).type(self.dtype)
            else:
                probabilities = torch.tensor([self.q, self.p, 1 - self.p - self.q]).to(self.device).type(self.dtype)
        elif location > 0:
            probabilities = torch.tensor([1 - self.p - self.q, self.p, self.q]).to(self.device).type(self.dtype)
        else:
            probabilities = torch.tensor([self.q, self.p, 1 - self.p - self.q]).to(self.device).type(self.dtype)

        if case == 2:
            probabilities[0] = 0
            probabilities /= probabilities.sum()
        elif case == 1:
            probabilities[2] = 0
            probabilities /=  probabilities.sum()

        return probabilities

    def odefunc_forward(self,t,x:AbstractElement,bounds = False, keys = None):
        out = x.clone()
        count = 0
        if bounds:
            bounds_list = []
        for i in range(len(self.odefunc)):
            if bounds and self.odefunc[i].needs_bounds:
                bounds_list.append(out.concretize())
            if i in self.requires_time:
                out = self.odefunc[i](t,out)
            elif self.use_lambda and self.odefunc[i].deepz_lambda != None:

                temp_keys = list(keys)
                temp_keys.append(count)
                temp_keys = tuple(temp_keys)

                lb,ub = out.concretize()

                if ((lb <0) * (ub>0)).sum() > 0:
                    dict_flag = True
                    if temp_keys in self.deepz_lambda_dict.keys():
                        self.odefunc[i].deepz_lambda = self.deepz_lambda_dict[temp_keys]
                    else:
                        self.odefunc[i].deepz_lambda = nn.Parameter(0.5* torch.ones(self.output_dim, dtype=out.dtype))
                        self.odefunc[i].deepz_lambda.to(self.device) 

                out = self.odefunc[i](out)

                if dict_flag:
                    #pdb.set_trace()
                    self.deepz_lambda_dict[temp_keys] = self.odefunc[i].deepz_lambda
                count += 1
            else:
                out = self.odefunc[i](out)
        if bounds:
            return out,bounds_list
        return out

    def rk4_step(self,x,t0,dt):
        k1 = self.odefunc_forward(t0,x)#k1 = f(t0,y0)
        temp = x.clone().add(k1*(dt/3))
        k2 = self.odefunc_forward(t0 + (dt/3),temp) #k2 = f(t0 + 1/3*dt,y0 + 1/3*dt*k1)

        temp = x.clone().add(k2*dt)
        temp = temp.add(k1*(dt*(-1)/3))
        k3 = self.odefunc_forward(t0 + (dt*2/3),temp) #k3 = f(t0 + 2/3*dt,y0  -1/3*k1*dt + k2*dt)

        temp = x.clone().add(k3*dt)
        temp = temp.add(k2*(dt*(-1)))
        temp = temp.add(k1*(dt))
        k4 = self.odefunc_forward(t0 + dt,temp) #k4 = f(t0 + 1*dt,y0 +dt*(k1-k2+k3))

        final = x.clone().add(0.125*dt *k1)
        final = final.add(0.125*dt *k4)
        final = final.add(0.375*dt *k2)
        final = final.add(0.375*dt *k3)
        
        return final

    def euler_step(self,x,t0,dt):
        k1 = self.odefunc_forward(t0,x)#k1 = f(t0,y0)
        #x_t+1 = x_t + dt* k1
        return x.clone().add(k1*dt)
    def _round(self,x):
        return x
        return torch.round(x,decimals=4)

    def dopri5_step(self,x,t0,dt,keys = None):
        bounds = {}

        temp_key = [*keys]
        temp_key.append("k1")
        k1,bounds['k1'] = self.odefunc_forward(t0,x,True,temp_key) #k1 = f(t0,y0)

        temp = x.clone().add(k1*self._round(dt/5))
        temp_key = [*keys]
        temp_key.append("k2")
        k2, bounds['k2']= self.odefunc_forward(t0 + self._round(dt/5),temp,True,temp_key) #k2 = f(t0 + 1/5*dt,y0 + 1/5*dt*k1)


        temp = x.clone().add(k2*self._round(9/40*dt))
        temp = temp.add(k1*self._round(3/40*dt))
        temp_key = [*keys]
        temp_key.append("k3")
        k3,bounds['k3']= self.odefunc_forward(t0 + self._round(dt*3/10),temp,True,temp_key) #k3 = f(t0 + 3/10*dt,y0  +9/40*k2*dt + +3/40*k2*dt)

        temp = x.clone().add(k1*self._round(dt*44/45))
        temp = temp.add(k2*self._round(dt*(-56)/15))
        temp = temp.add(k3*self._round(dt*32/9))
        temp_key = [*keys]
        temp_key.append("k4")
        k4,bounds['k4'] = self.odefunc_forward(t0 + self._round(dt*4/5), temp,True,temp_key) #k4 = f(t0 + 4/5*dt,y0 + 44/45*k1 + -56/15*k2 + 32/9k3)

        temp = x.clone().add(k1*self._round(dt*19372/6561))
        temp = temp.add(k2*self._round(dt*(-25360)/2187))
        temp = temp.add(k3*self._round(dt*64448/6561))
        temp = temp.add(k4*self._round(dt*(-212)/729))
        temp_key = [*keys]
        temp_key.append("k5")
        k5,bounds['k5'] = self.odefunc_forward(t0 + self._round(dt*8/9), temp,True,temp_key) #k5 = f(t0 + 8/9*dt,y0 + 19372/6561*k1 + -25360/2187*k2 + 64448/6561*k3 + -212/729*k4)

        temp = x.clone().add(k1*self._round(dt*9017/3168))
        temp = temp.add(k2*self._round(dt*(-355)/33))
        temp = temp.add(k3*self._round(dt*46732/5247))
        temp = temp.add(k4*self._round(dt* 49/176))
        temp = temp.add(k5*self._round(dt*(-5103)/18656))
        temp_key = [*keys]
        temp_key.append("k6")
        k6,bounds['k6'] = self.odefunc_forward(t0 + self._round(dt), temp,True,temp_key) #k6 = f(t0 + dt,y0 + blabla*k1 + ... + blabla * j5)

        y = x.clone().add(k1*self._round(dt*35/384))
        y = y.add(k3*self._round(dt*500/1113))
        y = y.add(k4 *self._round(dt* 125/192))
        y = y.add(k5 *self._round(dt*(-2187)/6784))
        y = y.add(k6*self._round(dt*(11)/84))
        
        temp_key = [*keys]
        temp_key.append("k7")
        k7 = self.odefunc_forward(t0 + self._round(dt), y,False,temp_key) 

        ####### THIS IS err CALCULATION ########
        err = (k1*self._round(dt*(35/384 - 1951 / 21600)))
        err = err.add(k3*self._round(dt*( 500 / 1113 - 22642 / 50085)))
        err = err.add(k4*self._round(dt*(125/192 - 451 / 720)))
        err = err.add(k5*self._round(dt*(12231 / 42400 - 2187/ 6784)))
        err = err.add(k6*self._round(dt*(11/84 - 649 / 6300)))
        err = err.add(k7*self._round(dt*(-1/60)))
        return y,err,bounds

    def bosh3_step(self,x,t0,dt):
        bounds = {}
        k1,bounds['k1'] = self.odefunc_forward(t0,x,True) #k1 = f(t0,y0)

        temp = x.clone().add(k1*self._round(dt/2))
        k2, bounds['k2']= self.odefunc_forward(t0 + self._round(dt/2),temp,True) #k2 = f(t0 + 1/2*dt,y0  +1/2*k1*dt )


        temp = x.clone().add(k2*self._round(3/4*dt))
        #temp = temp.add(k1*self._round(0*dt))
        k3,bounds['k3']= self.odefunc_forward(t0 + self._round(dt*3/4),temp,True) #k3 = f(t0 + 3/4*dt,y0  +3/4*k2*dt + +0*k1*dt)

        temp = x.clone().add(k1*self._round(dt*2/9))
        temp = temp.add(k2*self._round(dt*(1/3)))
        temp = temp.add(k3*self._round(dt*4/9))
        y = temp.clone()
        k4,bounds['k4'] = self.odefunc_forward(t0 + self._round(dt*1), temp,True) #k4 = f(t0 + 1*dt,y0 + 2/9*k1 + 1/3*k2 + 4/9k3)


        ####### THIS IS err CALCULATION ########
        err = (k1*self._round(dt*(2/9 - 7 / 24)))
        err = err.add(k2*self._round(dt*( 1 / 3 - 1 / 4)))
        err = err.add(k3*self._round(dt*(4/9 - 1 / 3)))
        err = err.add(k4*self._round(dt*(0 - 1/ 8)))
        return y,err,bounds


    @classmethod
    def from_concrete_layer(
        cls, layer, input_dim: Tuple[int, ...], running_mean_step = None, odesolver = None
    ) -> "ODEBlock_A":

        if str(type(layer)) =="<class 'latent_model._ODEfunc'>":
            abstract_layer = cls(
                layer.method,
                layer.integration_time[-1],
                layer.device
            )
            abstract_layer.requires_time = []
            abstract_layer.running_mean_step = running_mean_step
            abstract_layer.mode = "training"

            abstract_layer.odefunc = Sequential.from_concrete_network(layer.layers,input_dim)
            abstract_layer.output_dim = abstract_layer.odefunc[-1].output_dim
            abstract_layer.needs_bounds = True
            abstract_layer.forward_computed = False

            abstract_layer.training = layer.training
            abstract_layer.odesolver = odesolver
            abstract_layer.dtype = layer.layers[0].bias.dtype
            return abstract_layer

        abstract_layer = cls(
            layer.method,
            layer.integration_time[-1],
            layer.odefunc.conv1._layer.bias.device
        )
        seq = []
        abstract_layer.requires_time = []
        temp = 0
        abstract_layer.running_mean_step = layer.running_mean_step
        seq.append(layer.odefunc.conv1)
        abstract_layer.requires_time.append(temp)
        temp += 1
        try:
            seq.append(layer.odefunc.norm1)
            abstract_layer.requires_time.append(temp)
            temp += 1
            abstract_layer.normalized = True
        except:
            abstract_layer.normalized = False

        seq.append(layer.odefunc.act1)
        temp += 1

        seq.append(layer.odefunc.conv2)
        abstract_layer.requires_time.append(temp)
        temp += 1

        if abstract_layer.normalized:
            seq.append(layer.odefunc.norm2)
            abstract_layer.requires_time.append(temp)
            temp += 1

        seq.append(layer.odefunc.act2)
        temp += 1

        abstract_layer.odefunc = Sequential.from_concrete_network(nn.Sequential(*seq),input_dim)
        abstract_layer.output_dim = abstract_layer.odefunc[-1].output_dim
        abstract_layer.needs_bounds = True
        abstract_layer.forward_computed = False

        abstract_layer.training = layer.training
        abstract_layer.odesolver = layer.odesolver
        abstract_layer.dtype = abstract_layer.odefunc[0]._layer.bias.dtype
        abstract_layer.mode = "training"

        return abstract_layer


class BatchNorm(concrete_layers.BatchNorm, AbstractModule):
    def __init__(self,num_features,device,n_max_tp,endtime):
        super(BatchNorm, self).__init__(num_features,device,n_max_tp,endtime)
        
        self.n_max_tp = n_max_tp
        self.device = device
        self.endtime = endtime
        self.num_features = num_features

    def forward(self,t:Union[AbstractElement, Tensor], x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            key= (self.timepoints - t).abs().argmin()
            return x.batch_norm(self.BatchNorm2d[key])
        
        key= (self.timepoints - t).abs().argmin()

        current_mean = self.BatchNorm2d[key].running_mean
        current_var = self.BatchNorm2d[key].running_var

        c = (self.BatchNorm2d[key].weight / torch.sqrt(self.BatchNorm2d[key].current_var + self.eps))
        b = (-current_mean * c + self.BatchNorm2d[key].bias)
        return x*c.view(x.shape[0],x.shape[1],1,1)+b.view(x.shape[0],x.shape[1],1,1)

    @classmethod
    def from_concrete_layer(
        cls, layer, input_dim: Tuple[int, ...]
    ) -> "BatchNorm":
        abstract_layer = cls(
            layer.num_features,
            layer.device,
            layer.n_max_tp,
            layer.endtime,
        )
        abstract_layer.timepoints = layer.timepoints
        abstract_layer.BatchNorm2d = Sequential.from_concrete_network(layer.BatchNorm2d,input_dim)

        abstract_layer.training = False
        abstract_layer.output_dim = input_dim
        abstract_layer.needs_bounds = False

        return abstract_layer


class Encoder_ODE_RNN(concrete_layers.Encoder_z0_ODE_RNN, AbstractModule):
    def __init__(self, latent_dim, input_dim,
        z0_dim = None, device = torch.device("cpu"),input_init= 7):

        super(Encoder_ODE_RNN, self).__init__(latent_dim, input_dim, z0_dim ,device,input_init)
        self.device = device
        self.z0_dim = z0_dim
        self.mode = "training"

    def forward(self,data, time_steps,init_data, run_backwards = False,break_points = None,next_stop = None):
        if isinstance(data, AbstractElement):
            if len(time_steps) == 1:
                print("SHould not happen in enc")
                exit()
            elif self.mode == "training":
                n_traj, n_tp, n_dims = data.size()

                if run_backwards:
                    t0 = time_steps[-1]
                    prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]
                else:
                    t0 = time_steps[0]
                    prev_t, t_i = time_steps[0] * 0,  time_steps[0]

                if self.hidden_init != None:
                    temp = self.hidden_init(init_data)
                else:
                    temp = init_data
                prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
                prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

                interval_length = time_steps[-1] - time_steps[0]

                minimum_step = interval_length 

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
                            #euler step of stepsize t_i -prev t
                            #NOTE: at the moment doesn't depend on T
                        else:
                            n_intermediate_tp = max(2, ((prev_t - t_i).abs() / minimum_step).int())
                            #do multiple steps for better accuracy, but don't do to small steps
                            time_points = torch.linspace(prev_t, t_i, n_intermediate_tp).to(self.device)
                        yi_ode  = self.z0_diffeq_solver(prev_y, time_points)

                    xi = data[:,i,:].unsqueeze(0)
                    yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
                    prev_y, prev_std = yi, yi_std        

                    if run_backwards:
                        prev_t, t_i = time_steps[i],  time_steps[i-1]
                    elif i < (len(time_steps)-1):
                        prev_t,t_i =  time_steps[i], time_steps[i+1]   

                means_z0 = yi
                std_z0 = yi_std
                temp= self.transform_z0( yi.cat((yi,yi_std),-1))
                dims = temp.shape[-1]// 2
                mean_z0, std_z0 = temp[:,:,0:dims],temp[:,:,dims:]
                std_z0 = std_z0.abs()
                return mean_z0,std_z0

            elif self.mode == "abstract":
                n_traj, n_tp, n_dims = data.size()

                if run_backwards:
                    t0 = time_steps[-1]
                    prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]
                else:
                    t0 = time_steps[0]
                    prev_t, t_i = time_steps[0] * 0,  time_steps[0]

                if self.hidden_init != None:
                    temp,init_bounds = self.hidden_init(init_data,True)
                else:
                    temp = init_data
                    init_bounds = []
                prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
                prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

                interval_length = time_steps[-1] - time_steps[0]

                minimum_step = interval_length 
                needed_bounds = [[] for i in range(len(time_steps))]

                if break_points == None:
                    starting = 0
                else:    
                    starting, prev_y, prev_std = break_points[-1]
                    temp_bounds = self.bounds[1]

                    for i in range(starting):
                        needed_bounds[i] = temp_bounds[i]
                        if i < (len(time_steps)-1):
                            prev_t,t_i =  time_steps[i], time_steps[i+1]


                time_points_iter = range(starting, len(time_steps))

                if run_backwards:
                    time_points_iter = reversed(time_points_iter)

                self.dicts = {}
                for i in time_points_iter:

                    if (prev_t - t_i).abs() <1e-8:
                        yi_ode = prev_y
                        ode_bounds = []
                        time_points = None
                    else:

                        if (prev_t - t_i).abs() < minimum_step:
                            time_points = torch.stack((prev_t, t_i)).to(self.device)
                        else:
                            n_intermediate_tp = max(2, ((prev_t - t_i).abs() / minimum_step).int())
                            #do multiple steps for better accuracy, but don't do to small steps
                            time_points = torch.linspace(prev_t, t_i, n_intermediate_tp).to(self.device)

                        yi_ode,ode_bounds  = self.z0_diffeq_solver(prev_y, time_points,True)
                    needed_bounds[i].append([ode_bounds,time_points])
                    xi = data[:,i,:].unsqueeze(0)

                    yyy = yi_ode.cat((yi_ode,prev_std),-1)
                    self.dicts["pre"] = yyy.squeeze(1).concretize()
                    
                    yi, yi_std, gru_bounds = self.GRU_update(yi_ode, prev_std, xi,True)


                    needed_bounds[i].append(gru_bounds)

                    prev_y, prev_std = yi, yi_std    

                    if next_stop != None:
                        if i == next_stop -1:
                            if break_points == None:
                                break_points = [[next_stop,prev_y,prev_std]]
                            else:
                                break_points.append([next_stop,prev_y,prev_std])
                            self.bounds = [init_bounds,needed_bounds]
                            return break_points, False


                    if run_backwards:
                        prev_t, t_i = time_steps[i],  time_steps[i-1]
                    elif i < (len(time_steps)-1):
                        prev_t,t_i =  time_steps[i], time_steps[i+1]       

                means_z0 = yi
                std_z0 = yi_std

                temp, transform_bounds= self.transform_z0( yi.cat((yi,yi_std),-1),True)
                dims = temp.shape[-1]// 2
                mean_z0, std_z0 = temp[:,:,0:dims],temp[:,:,dims:]
                std_z0 = std_z0.abs()

                self.bounds = [init_bounds,needed_bounds,transform_bounds]
                return mean_z0,std_z0

        else:
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
            else:

                if run_backwards:
                    t0 = time_steps[-1]
                    prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]
                else:
                    t0 = time_steps[0]
                    prev_t, t_i = time_steps[0] * 0,  time_steps[0]


                if (data.type() == 'torch.DoubleTensor' or data.type() == 'torch.cuda.DoubleTensor') and self.hidden_init != None:
                    temp = self.hidden_init(init_data.double())
                elif self.hidden_init != None:
                    temp = self.hidden_init(init_data.float())
                else:
                    temp = torch.zeros((data.shape[0],2* self.latent_dim)).to(data.device)

                prev_y = temp[:,0:self.latent_dim].unsqueeze(0)
                prev_std = temp[:,self.latent_dim:].abs().unsqueeze(0)

                interval_length = time_steps[-1] - time_steps[0]
                minimum_step = interval_length 
                
                # Run ODE backwards and combine the y(t) estimates using gating
                time_points_iter = range(0, len(time_steps))
                if run_backwards:
                    time_points_iter = reversed(time_points_iter)

                for i in time_points_iter:
                    if (prev_t - t_i).abs() <1e-8:
                        #only possible at i == 0, h0 no ode just init
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

                last_yi, last_yi_std = yi, yi_std

            means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
            std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

            temp= self.transform_z0( torch.cat((means_z0, std_z0), -1))
            dims = temp.shape[-1]// 2
            mean_z0, std_z0 = temp[:,:,0:dims],temp[:,:,dims:]
            std_z0 = std_z0.abs()
            return mean_z0,std_z0

    @classmethod
    def from_concrete_layer(
        cls, layer: concrete_layers.Encoder_z0_ODE_RNN, input_dim: Tuple[int, ...]
    ) -> "Conv2d":
        abstract_layer = cls(
            latent_dim = layer.latent_dim,
            input_dim = layer.input_dim,
            z0_dim = layer.z0_dim,
            device= layer.device,
            input_init= layer.input_init
        )
        abstract_layer.hidden_init_dim = layer.input_init
        if abstract_layer.hidden_init_dim > 0:
            abstract_layer.hidden_init = Sequential_with_bounds(Sequential.from_concrete_network(layer.hidden_init,(layer.input_init)))
        else: 
            abstract_layer.hidden_init = None

        abstract_layer.transform_z0 = Sequential_with_bounds(Sequential.from_concrete_network(layer.transform_z0,(2 * layer.latent_dim)))
        abstract_layer.z0_diffeq_solver = DiffeqSolver.from_concrete_layer(layer.z0_diffeq_solver,(layer.input_dim))

        abstract_layer.GRU_update = GRU_unit.from_concrete_layer(layer.GRU_update,(layer.latent_dim * 2 + layer.input_dim,))
        
        abstract_layer.output_dim = (1,1,layer.latent_dim)
        abstract_layer.dim = input_dim
        abstract_layer.needs_bounds = True
        abstract_layer.requires_time = False

        return abstract_layer


class DiffeqSolver(concrete_layers.DiffeqSolver, AbstractModule):
    def __init__(self, input_dim, ode_func, method,latents, odeint_rtol = 0, odeint_atol = 5e-3, device = torch.device("cpu")):
        
        super(DiffeqSolver, self).__init__(input_dim, ode_func, method,latents)
        self.method = method
        self.latents = latents      
        self.device = device
        self.mode = "training"
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.step_size = None


    def forward(self,first_point, time_steps_to_predict, return_bounds = False, mask = None):
        if isinstance(first_point, AbstractElement) and self.mode == "training":
            if self.met == "euler":
                out = first_point.clone()
                bounds_list = []
                for i in range(len(time_steps_to_predict)-1):
                    dt = time_steps_to_predict[i+1] - time_steps_to_predict[i]
                    out,bounds = self.euler_step(out,time_steps_to_predict[i],dt,False)
                    bounds_list.append(bounds)

                return out

            elif self.met in ["dopri5_0.005_2a","dopri5_0.01_2a"]:
                self.ode_func_train.mode = "training"


                temp_lower = torch.zeros_like(first_point.concretize()[0]).unsqueeze(2).repeat([1,1,len(time_steps_to_predict),1])
                temp_upper = temp_lower.clone()
                for i in range(len(time_steps_to_predict)):
                    idx = torch.where(mask[:,i].sum(1) > 0)[0]
                    inputs = first_point[0,idx]
                    #we just use output of enc as output
                    if time_steps_to_predict[i] == 0:
                        lower,upper = inputs.concretize()
                        temp_lower[0,idx,i] += lower
                        temp_upper[0,idx,i] += upper
                        continue
                    self.ode_func_train.integration_time = time_steps_to_predict[i]
                    #needs to be pop and not indexed, bc if 0 in timepoints we would have issue
                    self.ode_func_train.last_step = self.step_sizes.pop(0)

                    self.ode_func_train.trajectory_path = self.trajectories.pop(0)
                    self.ode_func_train.states = self.states.pop(0)
                    out = self.ode_func_train(inputs)
                    lower,upper = out.concretize()

                    temp_lower[0,idx,i] += lower
                    temp_upper[0,idx,i] += upper
                return HybridZonotope.construct_from_bounds(min_x= temp_lower.permute(1,2,0,3), max_x=temp_upper.permute(1,2,0,3), domain="box",dtype = temp_lower.dtype)

            elif self.met == "euler_2":
                self.ode_func_train.mode = "training"
                temp_lower = torch.zeros_like(first_point.concretize()[0]).unsqueeze(2).repeat([1,1,len(time_steps_to_predict),1])
                temp_upper = temp_lower.clone()
                for i in range(len(time_steps_to_predict)):
                    idx = torch.where(mask[:,i].sum(1) > 0)[0]
                    inputs = first_point[0:1,idx]
                    #we just use output of enc as output
                    if time_steps_to_predict[i] == 0:
                        lower,upper = inputs.concretize()
                        temp_lower[0:1,idx,i] += lower
                        temp_upper[0:1,idx,i] += upper
                        continue
                    self.ode_func_train.integration_time = time_steps_to_predict[i]
                    self.ode_func_train.dt = time_steps_to_predict[i] / 2

                    out = self.ode_func_train(inputs)
                    lower,upper = out.concretize()
                    temp_lower[0:1,idx,i] += lower
                    temp_upper[0:1,idx,i] += upper
                return HybridZonotope.construct_from_bounds(min_x= temp_lower.permute(1,2,0,3), max_x=temp_upper.permute(1,2,0,3), domain="box",dtype = temp_lower.dtype)


        elif isinstance(first_point, AbstractElement) and (self.mode == "abstract" or return_bounds == True):
            if self.met == "euler":
                out = first_point.clone()
                bounds_list = []
                for i in range(len(time_steps_to_predict)-1):
                    dt = time_steps_to_predict[i+1] - time_steps_to_predict[i]
                    out,bounds = self.euler_step(out,time_steps_to_predict[i],dt,True)
                    bounds_list.append(bounds)
                return out, bounds_list

            if self.met == "euler_2":
                
                out = first_point.clone()
                #assumesonly one timepoint
                self.ode_func_train.mode = "abstract"
                
                self.ode_func_train.integration_time = time_steps_to_predict[0]
                self.ode_func_train.update_bounds(first_point.concretize())
                out = self.ode_func_train(first_point,True)

                return out

            elif self.met in ["dopri5_0.005_2a","dopri5_0.01_2a"]:
                self.ode_func_train.mode = "abstract"
                temp_lower = torch.zeros_like(first_point.concretize()[0]).unsqueeze(2).repeat([1,1,len(time_steps_to_predict),1])
                temp_upper = temp_lower.clone()
                running_mean_step = self.ode_func_train.running_mean_step.data.clone()
                for i in range(len(time_steps_to_predict)):
                    idx = torch.where(mask[:,i].sum(1) > 0)[0]
                    inputs = first_point[0,idx]
                    #we just use output of enc as output
                    if time_steps_to_predict[i] == 0:
                        lower,upper = inputs.concretize()
                        temp_lower[0,idx,i] += lower
                        temp_upper[0,idx,i] += upper
                        self.ode_func_train.graph = None
                        continue
                    self.ode_func_train.integration_time = time_steps_to_predict[i]
                    #needs to be pop and not indexed, bc if 0 in timepoints we would have issue
                    self.ode_func_train.last_step = None
                    self.ode_func_train.trajectory_path = None
                    self.ode_func_train.states = None
                    self.ode_func_train.update_bounds(inputs.concretize())
                    self.ode_func_train.running_mean_step.data = torch.clamp(running_mean_step,0.0,time_steps_to_predict[i])
                    out = self.ode_func_train(inputs,True)
                    lower,upper = out.concretize()
                    temp_lower[0,idx,i] += lower
                    temp_upper[0,idx,i] += upper
                self.ode_func_train.running_mean_step.data = running_mean_step
                return HybridZonotope.construct_from_bounds(min_x= temp_lower.permute(1,2,0,3), max_x=temp_upper.permute(1,2,0,3), domain="box",dtype = temp_lower.dtype)

        else:
            #CONCRETE DIFFEQSOLVER FWD PASS 
            n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
            n_dims = first_point.size()[-1]
            if self.met == "dopri5":
                    if 0 not in time_steps_to_predict:
                        time_steps_to_predict = torch.cat((torch.zeros_like(time_steps_to_predict[0:1]),time_steps_to_predict),0)
                        temp_flag = True
                    else:
                        temp_flag = False
                    pred_y = self.odesolver(self.ode_func, first_point[0], time_steps_to_predict, 
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
                    if temp_flag:
                        pred_y = pred_y[0][1:].permute(1,2,0,3)
                    else:
                        pred_y = pred_y[0].permute(1,2,0,3)

            elif self.met == "euler":
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
                states = []
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
                        states.append(self.liste[-3])
                    else:
                        out,self.liste = self.odesolver(self.ode_func, inputs, to_predict, method = self.ode_method,
                            rtol= self.odeint_rtol,atol= self.odeint_atol,step_size= self.step_size,
                            adaptive_step_factor = self.adaptive_step_factor,running_mean_step = torch.clamp(self.running_mean_step.data ,0.0,to_predict[-1]))
                        step_sizes.append(torch.clamp(self.running_mean_step.data,0.0,to_predict[-1]))
                        trajectories.append(self.liste[-4])
                        states.append(self.liste[-3])

                    out_data[0,idx,i] += out[1]

                self.step_sizes = step_sizes
                self.trajectories = trajectories
                self.states = states
                if self.training:
                    self.running_mean_step.data = torch.clamp((1-0.1)* self.running_mean_step.data + 0.1 * min(step_sizes),0,1)

                pred_y = out_data.permute(1,2,0,3)

            return pred_y 

    def euler_step(self,x,t0,dt,return_bounds):
        k1,bounds = self.ode_func(t0,x,return_bounds)#k1 = f(t0,y0)
        return x.clone().add(k1*dt),bounds

    def ode_func(self,t,x,return_bounds = False):
        if isinstance(x,AbstractElement) and return_bounds:
            out = x.clone()
            bounds_list = []
            for i in range(len(self.ode_func_f)):
                if self.ode_func_f[i].needs_bounds:
                    bounds_list.append(out.concretize())
                if self.ode_func_f[i].requires_time:
                    out = self.ode_func_f[i](t,out)
                else:
                    out = self.ode_func_f[i](out)
            return out,bounds_list
        elif isinstance(x,AbstractElement):
            return self.ode_func_f(x), None
        else:
            return self.ode_func_f(t,x)
    @classmethod
    def from_concrete_layer(
        cls, layer: concrete_layers.DiffeqSolver, input_dim: Tuple[int, ...]
    ) -> "Conv2d":
        abstract_layer = cls( input_dim = layer.input_dim,
            ode_func= layer.ode_func,
            method = layer.ode_method,
            latents = layer.latents,
            odeint_rtol = layer.odeint_rtol, 
            odeint_atol = layer.odeint_atol,
            device= layer.device
        )

        abstract_layer.ode_method = layer.ode_method
        abstract_layer.met = layer.met
        abstract_layer.adaptive_step_factor = layer.adaptive_step_factor
        abstract_layer.running_mean_step = layer.running_mean_step
        layer.ode_func.method = layer.met
        layer.ode_func.integration_time = torch.tensor([0.0,1.0]).to(layer.device).type(layer.ode_func.layers[0].bias.dtype)

        layer.ode_func.device = layer.device

        abstract_layer.ode_func_train = ODEBlock_A.from_concrete_layer(layer.ode_func,(layer.input_dim),layer.running_mean_step,layer.odesolver)
        abstract_layer.ode_func_f = Sequential.from_concrete_network(layer.ode_func.layers,(layer.input_dim))
        abstract_layer.odesolver = layer.odesolver
        abstract_layer.output_dim = (input_dim)
        abstract_layer.dim = input_dim
        abstract_layer.needs_bounds = False
        abstract_layer.requires_time = False

        return abstract_layer



class GRU_unit(concrete_layers.GRU_unit, AbstractModule):
    def __init__(self,latent_dim,input_dim,device = torch.device("cpu")):
        
        super(GRU_unit, self).__init__(latent_dim,input_dim)
        self.device = device
        self.input_dim = input_dim

    def forward(self, y_mean, y_std, x, return_bounds = False,masked_update = True):
        if isinstance(y_mean, AbstractElement):
            n_data_dims = self.input_dim//2
            x,mask = x[:,:,0:n_data_dims],x[:,:,n_data_dims:]
            lb,ub = mask.concretize()
            mask = lb
            y_concat = y_mean.cat((y_mean.clone(), y_std.clone(), x),-1)
            if return_bounds:
                bounds_dict = {}
                update_gate,bounds_dict["update_bounds"] = self.update_gate(y_concat.clone(),return_bounds)
                reset_gate,bounds_dict["reset_bounds"]  = self.reset_gate(y_concat.clone(),return_bounds)
            else:
                update_gate = self.update_gate(y_concat)
                reset_gate= self.reset_gate(y_concat)

            y_m = y_mean.prod(reset_gate)
            y_s = y_std.prod(reset_gate)

            concat = y_mean.cat((y_m, y_s, x),-1)

            if return_bounds:
                temp,bounds_dict["state_bounds"] = self.new_state_net(concat,return_bounds)
            else:
                temp= self.new_state_net(concat)

            dims = temp.shape[-1]// 2
            new_state, new_state_std  = temp[:,:,0:dims],temp[:,:,dims:]
            if return_bounds:
                bounds_dict["state_abs_box"] = new_state_std.squeeze(1).concretize()

            new_state_std = new_state_std.abs()

            update_gate_2 = (-1 * update_gate).add(1)

            if return_bounds:
                y = y_mean.cat((y_mean,y_std),-1)
                bounds_dict["y_box"] = y.squeeze(1).concretize()
                g_y = new_state.cat((new_state,new_state_std),-1)
                bounds_dict["state_box"] = g_y.squeeze(1).concretize()

                reset_lb,reset_ub = reset_gate.squeeze(1).concretize()
                reset_lb,reset_ub = torch.clamp(reset_lb,min=0.0,max=1.0), torch.clamp(reset_ub,min=0.0,max=1.0)
                bounds_dict["reset_box"] = (torch.cat((reset_lb,reset_lb),-1),torch.cat((reset_ub,reset_ub),-1))

                update_lb,update_ub = update_gate_2.squeeze(1).concretize()
                update_lb,update_ub = torch.clamp(update_lb,min=0.0,max=1.0), torch.clamp(update_ub,min=0.0,max=1.0)
                bounds_dict["inverse_update_box"] = (torch.cat((update_lb,update_lb),-1),torch.cat((update_ub,update_ub),-1))

                update_lb,update_ub = update_gate.squeeze(1).concretize() 
                update_lb,update_ub = torch.clamp(update_lb,min=0.0,max=1.0), torch.clamp(update_ub,min=0.0,max=1.0)
                bounds_dict["update_box"] = (torch.cat((update_lb,update_lb),-1),torch.cat((update_ub,update_ub),-1))

            new_y = new_state.prod(update_gate_2) + y_mean.prod(update_gate)
            new_y_std =  new_state_std.prod(update_gate_2) + y_std.prod(update_gate)

            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()
            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std


            if return_bounds:
                return new_y,new_y_std,bounds_dict
            return new_y,new_y_std



        else:
            n_data_dims = self.input_dim//2
            x,mask = x[:,:,0:n_data_dims],x[:,:,n_data_dims:]
            y_concat = torch.cat([y_mean, y_std, x], -1)
            update_gate = self.update_gate(y_concat)
            reset_gate = self.reset_gate(y_concat)
            concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

            temp= self.new_state_net(concat)
            dims = temp.shape[-1]// 2
            new_state, new_state_std  = temp[:,:,0:dims],temp[:,:,dims:]
            new_state_std = new_state_std.abs()

            new_y = (1-update_gate) * new_state + update_gate * y_mean
            new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

            if masked_update:
                mask = (torch.sum(mask, -1, keepdim = True) > 0).float()
                new_y = mask * new_y + (1-mask) * y_mean
                new_y_std = mask * new_y_std + (1-mask) * y_std

            return new_y,new_y_std 

    @classmethod
    def from_concrete_layer(
        cls, layer: concrete_layers.GRU_unit, input_dim: Tuple[int, ...]
    ) -> "Conv2d":
        abstract_layer = cls(latent_dim = layer.latent_dim, input_dim = layer.input_dim, device = layer.device)

        abstract_layer.update_gate = Sequential_with_bounds(Sequential.from_concrete_network(layer.update_gate,(input_dim)))
        abstract_layer.reset_gate = Sequential_with_bounds(Sequential.from_concrete_network(layer.reset_gate,(input_dim)))
        abstract_layer.new_state_net = Sequential_with_bounds(Sequential.from_concrete_network(layer.new_state_net,(input_dim)))

        abstract_layer.output_dim = (layer.latent_dim * 2)
        abstract_layer.dim = input_dim
        abstract_layer.needs_bounds = True
        abstract_layer.requires_time = False
        return abstract_layer



class Sequential_with_bounds(AbstractModule):
    def __init__(self,net,device = torch.device("cpu")):
        
        super(Sequential_with_bounds, self).__init__(net)
        self.device = device
        self.net = net
        self.needs_bounds = True
        self.requires_time = False

    def forward(self, x, return_bounds= False):
        if isinstance(x, AbstractElement) and return_bounds:
            out = x.clone()
            bounds_list = []
            for i in range(len(self.net)):
                if self.net[i].needs_bounds:
                    bounds_list.append(out.concretize())
                out = self.net[i](out)
            return out,bounds_list
        return self.net(x)
    @classmethod
    def from_concrete_layer(
        cls, layer, input_dim: Tuple[int, ...]
    ) -> "Conv2d":

        abstract_layer = cls(net= layer,  device = layer[0].bias.device)
        abstract_layer.net = Sequential.from_concrete_network(layer,input_dim)

        abstract_layer.output_dim = abstract_layer.net[-1].output_dim
        abstract_layer.dim = input_dim
        abstract_layer.needs_bounds = True
        abstract_layer.requires_time = False
        return abstract_layer

