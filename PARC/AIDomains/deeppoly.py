import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union
from torch import Tensor
import pdb

import copy

from PARC.AIDomains.abstract_layers import Normalization, Linear, ReLU, Conv2d, Flatten, GlobalAvgPool2d, AvgPool2d, Upsample, _BatchNorm, Bias, Scale, ResBlock, Sequential, ConcatConv, ODEBlock_A, GroupNorm, BatchNorm, Sigmoid
from .ai_util import get_neg_pos_comp


class DeepPoly:
    def __init__(self, x_l_coef: Optional[Tensor]=None, x_u_coef: Optional[Tensor]=None, x_l_bias: Optional[Tensor]=None,
                 x_u_bias: Optional[Tensor]=None, expr_coef: Optional[Tensor]=None) -> None:
        '''
        expr_coeff is used for the initialization to define the linear expression to be bounded
        '''
        if expr_coef is None and (x_l_coef is None or x_u_coef is None):
            return
        assert expr_coef is None or isinstance(expr_coef, torch.Tensor)
        self.device = x_l_coef.device if expr_coef is None else expr_coef.device

        self.x_l_coef = expr_coef if x_l_coef is None else x_l_coef
        self.x_u_coef = expr_coef if x_u_coef is None else x_u_coef
        self.x_l_bias = torch.tensor(0) if x_l_bias is None else x_l_bias
        self.x_u_bias = torch.tensor(0) if x_u_bias is None else x_u_bias

    def clone(self) -> "DeepPoly":
        return DeepPoly(self.x_l_coef.clone(), self.x_u_coef.clone(), self.x_l_bias.clone(), self.x_u_bias.clone())

    def detach(self) -> "DeepPoly":
        x_l_coef = self.x_l_coef.detach()
        x_u_coef = self.x_u_coef.detach()
        x_l_bias = self.x_l_bias.detach()
        x_u_bias = self.x_u_bias.detach()
        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_linear(self, weight: Tensor, bias: Tensor) -> "DeepPoly":
        x_l_bias = self.x_l_bias + (0 if bias is None else self.x_l_coef.matmul(bias))
        x_u_bias = self.x_u_bias + (0 if bias is None else self.x_u_coef.matmul(bias))
        if len(weight.shape) == 0:
            x_l_coef = self.x_l_coef * weight
            x_u_coef = self.x_u_coef * weight
        else:
            x_l_coef = self.x_l_coef.matmul(weight)
            x_u_coef = self.x_u_coef.matmul(weight)

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_bias(self, bias: Tensor) -> "DeepPoly":
        view_dim = (1, 1) + (bias.shape)

        x_l_bias = self.x_l_bias + (self.x_l_coef*bias.view(view_dim)).sum(tuple(range(2-self.x_l_coef.dim(),0)))
        x_u_bias = self.x_u_bias + (self.x_u_coef*bias.view(view_dim)).sum(tuple(range(2-self.x_l_coef.dim(),0)))
        return DeepPoly(self.x_l_coef, self.x_u_coef, x_l_bias, x_u_bias)

    def dp_scale(self, scale: Tensor) -> "DeepPoly":
        view_dim = (1, 1) + (scale.shape)
        x_l_coef = self.x_l_coef*scale.view(view_dim)
        x_u_coef = self.x_u_coef*scale.view(view_dim)
        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_add(self, other: "DeepPoly") -> "DeepPoly":
        x_l_coef = self.x_l_coef + other.x_l_coef
        x_u_coef = self.x_u_coef + other.x_u_coef
        x_l_bias = self.x_l_bias + other.x_l_bias
        x_u_bias = self.x_u_bias + other.x_u_bias
        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_global_avg_pool2d(self, preconv_wh: Union[Tensor, torch.Size]) -> "DeepPoly":
        sz = self.x_l_coef.shape
        input_spatial_size = np.prod(preconv_wh[-2:])
        dtype=self.x_l_coef.dtype
        device=self.x_l_coef.device

        x_l_coef = self.x_l_coef * torch.ones((1,1,1,*preconv_wh[-2:]), dtype=dtype, device=device)/input_spatial_size
        x_u_coef = self.x_u_coef * torch.ones((1,1,1,*preconv_wh[-2:]), dtype=dtype, device=device)/input_spatial_size

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_avg_pool2d(self, preconv_wh: Union[Tensor, torch.Size], kernel_size: Union[Tuple[int,int],int],
                      stride: Union[Tuple[int,int],int], padding: Union[Tuple[int,int],int]) -> "DeepPoly":
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
        dtype = self.x_l_coef.dtype
        device = self.x_l_coef.device

        w_padding = (preconv_wh[1] + 2 * padding[0] - kernel_size[0]) % stride[0]
        h_padding = (preconv_wh[2] + 2 * padding[1] - kernel_size[1]) % stride[1]
        output_padding = (w_padding, h_padding)

        sz = self.x_l_coef.shape

        weight = 1/(np.prod(kernel_size)) * torch.ones((preconv_wh[0],1,*kernel_size), dtype=dtype, device=device)

        new_x_l_coef = F.conv_transpose2d(self.x_l_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, preconv_wh[0], 1)
        new_x_u_coef = F.conv_transpose2d(self.x_u_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, preconv_wh[0], 1)
        x_l_coef = new_x_l_coef.view((sz[0], sz[1], *new_x_l_coef.shape[1:]))
        x_u_coef = new_x_u_coef.view((sz[0], sz[1], *new_x_u_coef.shape[1:]))

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_normalize(self, mean: Tensor, sigma: Tensor) -> "DeepPoly":
        req_shape = [1] * self.x_l_coef.dim()
        req_shape[2] = mean.shape[1]
        assert req_shape[2] ==3 or req_shape[2] == 1
        x_l_bias = (self.x_l_bias + (self.x_l_coef * (-mean / sigma).view(req_shape)).view(*self.x_l_coef.size()[:2], -1).sum(2))
        x_u_bias = (self.x_u_bias + (self.x_u_coef * (-mean / sigma).view(req_shape)).view(*self.x_u_coef.size()[:2], -1).sum(2))

        x_l_coef = self.x_l_coef / sigma.view(req_shape)
        x_u_coef = self.x_u_coef / sigma.view(req_shape)

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_relu(self, bounds: Tuple[Tensor], it: int, dp_lambda:Optional[Tensor]=None) -> "DeepPoly":
        x_lb, x_ub = bounds

        lambda_l = torch.where(x_ub < -x_lb, torch.zeros(x_lb.size()).to(x_lb.device).type(x_lb.type()), torch.ones(x_lb.size()).to(x_lb.device).type(x_lb.type()))
        lambda_u = x_ub / (x_ub - x_lb + 1e-15)

        if dp_lambda is not None:
            lambda_l = dp_lambda.view(lambda_l.shape)
        if it == 17:
            lambda_l = lambda_l * 0
        if it == 18:
            lambda_l = lambda_l * 0 + 1

        # stably inactive
        lambda_l = torch.where(x_ub < 0, torch.zeros(x_lb.size()).to(x_lb.device).type(x_lb.type()), lambda_l)
        lambda_u = torch.where(x_ub < 0, torch.zeros(x_ub.size()).to(x_ub.device).type(x_lb.type()), lambda_u)

        # stably active
        lambda_l = torch.where(x_lb > 0, torch.ones(x_lb.size()).to(x_lb.device).type(x_lb.type()), lambda_l)
        lambda_u = torch.where(x_lb > 0, torch.ones(x_ub.size()).to(x_ub.device).type(x_lb.type()), lambda_u)

        mu_l = torch.zeros(x_lb.size()).to(x_lb.device).type(x_lb.type())
        mu_u = torch.where((x_lb < 0) & (x_ub > 0), -x_ub * x_lb / (x_ub - x_lb + 1e-15),
                           torch.zeros(x_lb.size()).to(x_lb.device).type(x_lb.type()))  # height of upper bound intersection with y axis

        lambda_l, lambda_u = lambda_l.unsqueeze(1), lambda_u.unsqueeze(1)
        mu_l, mu_u = mu_l.unsqueeze(1), mu_u.unsqueeze(1)

        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)

        x_l_coef = pos_x_l_coef * lambda_l + neg_x_l_coef * lambda_u
        new_x_l_bias = pos_x_l_coef * mu_l + neg_x_l_coef * mu_u
        x_u_coef = pos_x_u_coef * lambda_u + neg_x_u_coef * lambda_l
        new_x_u_bias = pos_x_u_coef * mu_u + neg_x_u_coef * mu_l

        if len(new_x_l_bias.size()) == 3:
            new_x_l_bias = new_x_l_bias.sum(2)
            new_x_u_bias = new_x_u_bias.sum(2)
        else:
            new_x_l_bias = new_x_l_bias.sum((2, 3, 4))
            new_x_u_bias = new_x_u_bias.sum((2, 3, 4))

        x_l_bias = self.x_l_bias + new_x_l_bias
        x_u_bias = self.x_u_bias + new_x_u_bias

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)


    def dp_sigmoid(self, bounds: Tuple[Tensor], it: int, dp_lambda:Optional[Tensor]=None) -> "DeepPoly":
        x_lb, x_ub = bounds

        sig_u = torch.sigmoid(x_ub)
        sig_l = torch.sigmoid(x_lb)
        lambda_1= (sig_u - sig_l)/(x_ub - x_lb + 1e-15)
        lambda_2 = torch.minimum(sig_u * (1 - sig_u ),sig_l * (1 - sig_l))

        lambda_l = torch.where(x_lb < 0, lambda_1, lambda_2)
        lambda_u = torch.where(x_ub > 0, lambda_2, lambda_1)

        mu_l = sig_l - lambda_l * x_lb
        mu_u = sig_u - lambda_u * x_ub

        lambda_l, lambda_u = lambda_l.unsqueeze(1), lambda_u.unsqueeze(1)
        mu_l, mu_u = mu_l.unsqueeze(1), mu_u.unsqueeze(1)

        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)

        x_l_coef = pos_x_l_coef * lambda_l + neg_x_l_coef * lambda_u
        new_x_l_bias = pos_x_l_coef * mu_l + neg_x_l_coef * mu_u
        x_u_coef = pos_x_u_coef * lambda_u + neg_x_u_coef * lambda_l
        new_x_u_bias = pos_x_u_coef * mu_u + neg_x_u_coef * mu_l

        if len(new_x_l_bias.size()) == 3:
            new_x_l_bias = new_x_l_bias.sum(2)
            new_x_u_bias = new_x_u_bias.sum(2)
        else:
            new_x_l_bias = new_x_l_bias.sum((2, 3, 4))
            new_x_u_bias = new_x_u_bias.sum((2, 3, 4))

        x_l_bias = self.x_l_bias + new_x_l_bias
        x_u_bias = self.x_u_bias + new_x_u_bias

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_conv(self, preconv_wh: Union[Tensor, torch.Size], weight: Tensor, bias: Tensor,
                stride: Union[Tuple[int,int],int], padding: Union[Tuple[int,int],int], groups: int,
                dilation: Union[Tuple[int,int],int]) -> "DeepPoly":
        kernel_wh = weight.shape[-2:]
        w_padding = (preconv_wh[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_wh[0] - 1)) % stride[0]
        h_padding = (preconv_wh[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_wh[1] - 1)) % stride[1]
        output_padding = (w_padding, h_padding)

        sz = self.x_l_coef.shape

        # process reference
        x_l_bias = self.x_l_bias + (0 if bias is None else (self.x_l_coef.sum((3, 4)) * bias).sum(2))
        x_u_bias = self.x_u_bias + (0 if bias is None else (self.x_u_coef.sum((3, 4)) * bias).sum(2))

        new_x_l_coef = F.conv_transpose2d(self.x_l_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_x_u_coef = F.conv_transpose2d(self.x_u_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, groups, dilation)
        x_l_coef = new_x_l_coef.view((sz[0], sz[1], *new_x_l_coef.shape[1:]))
        x_u_coef = new_x_u_coef.view((sz[0], sz[1], *new_x_u_coef.shape[1:]))

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_concat_conv(self, preconv_wh: Union[Tensor, torch.Size], weight: Tensor, bias: Tensor, time:Union[Tensor,float],
                stride: Union[Tuple[int,int],int], padding: Union[Tuple[int,int],int], groups: int,
                dilation: Union[Tuple[int,int],int]) -> "DeepPoly":
        kernel_wh = weight.shape[-2:]
        w_padding = (preconv_wh[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_wh[0] - 1)) % stride[0]
        h_padding = (preconv_wh[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_wh[1] - 1)) % stride[1]
        output_padding = (w_padding, h_padding)
        sz = self.x_l_coef.shape
        w1 = weight[:,0,:,:]
        w1 = w1.view(w1.shape[0],1,w1.shape[1],w1.shape[2])
        weight = weight[:,1:,:,:]

        temp = torch.ones((1,1,*preconv_wh[1:])).type(self.x_l_coef.type()) * time
        bias1 = F.conv2d(temp,w1,None, stride=stride, padding=padding, groups=groups, dilation=dilation)

        # process reference
        bias = bias1 + bias.view(1, -1, 1, 1)
        x_l_bias = self.x_l_bias + (0 if bias is None else (self.x_l_coef * bias)).sum((2,3,4))
        x_u_bias = self.x_u_bias + (0 if bias is None else (self.x_u_coef * bias)).sum((2,3,4))

        new_x_l_coef = F.conv_transpose2d(self.x_l_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_x_u_coef = F.conv_transpose2d(self.x_u_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, groups, dilation)
        x_l_coef = new_x_l_coef.view((sz[0], sz[1], *new_x_l_coef.shape[1:]))
        x_u_coef = new_x_u_coef.view((sz[0], sz[1], *new_x_u_coef.shape[1:]))

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_flatten(self, input_size: Union[torch.Size, List[int]]) -> "DeepPoly":

        x_l_coef = self.x_l_coef.view(*self.x_l_coef.size()[:3], *input_size)
        x_u_coef = self.x_u_coef.view(*self.x_u_coef.size()[:3], *input_size)

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_concretize(self, bounds: Optional[Tuple[Tensor]]=None, abs_input: Optional["HybridZonotope"]=None) -> "DeepPoly":
        assert not (bounds is None and abs_input is None)
        if abs_input is not None and abs_input.domain == "zono":
            abs_lb = abs_input.flatten().linear(self.x_l_coef.view(-1, abs_input.head.numel()), bias=self.x_l_bias.flatten()).view(self.x_l_bias.shape).concretize()[0]
            abs_ub = abs_input.flatten().linear(self.x_u_coef.view(-1, abs_input.head.numel()), bias=self.x_u_bias.flatten()).view(self.x_l_bias.shape).concretize()[1]
            return abs_lb, abs_ub
        if bounds is None:
            bounds = abs_input.concretize()
            
        lb_x, ub_x = bounds
        lb_x, ub_x = lb_x.unsqueeze(1), ub_x.unsqueeze(1)
        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)
        x_l_bias = self.x_l_bias + (pos_x_l_coef * lb_x + neg_x_l_coef * ub_x).view(lb_x.size()[0], self.x_l_coef.size()[1], -1).sum(2)
        x_u_bias = self.x_u_bias + (pos_x_u_coef * ub_x + neg_x_u_coef * lb_x).view(lb_x.size()[0], self.x_l_coef.size()[1], -1).sum(2)

        return x_l_bias, x_u_bias

    def dp_backsub(self, other) -> "DeepPoly":
        shp_self = self.x_u_coef.shape
        shp_other = other.x_u_coef.shape
        temp = 1
        for x in shp_self[2:]:
            temp *= x
        assert temp == shp_other[1]

        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef.view(*shp_self[:2],temp))
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef.view(*shp_self[:2],temp))

        x_l_coef = (torch.matmul(neg_x_l_coef, other.x_u_coef.view(*shp_other[:2],temp)) + torch.matmul(pos_x_l_coef, other.x_l_coef.view(*shp_other[:2],temp))).view(*shp_self)
        x_u_coef = (torch.matmul(neg_x_u_coef, other.x_l_coef.view(*shp_other[:2],temp)) + torch.matmul(pos_x_u_coef, other.x_u_coef.view(*shp_other[:2],temp))).view(*shp_self)

        if len(other.x_l_bias.shape) == 0:
            x_l_bias = self.x_l_bias
            x_u_bias = self.x_u_bias
        else:
            x_l_bias = self.x_l_bias + (pos_x_l_coef.matmul(other.x_l_bias.view(-1)) + neg_x_l_coef.matmul(other.x_u_bias.view(-1)))
            x_u_bias = self.x_u_bias + (pos_x_u_coef.matmul(other.x_u_bias.view(-1)) + neg_x_u_coef.matmul(other.x_l_bias.view(-1)))

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_upsample(self, pre_sample_size:Union[Tensor, torch.Size], mode:str, align_corners:bool):
        sz = self.x_l_coef.shape

        new_x_l_coef = F.interpolate(self.x_l_coef.view((-1, *sz[-3:])), size=pre_sample_size, mode=mode,
                                     align_corners=align_corners)
        new_x_u_coef = F.interpolate(self.x_u_coef.view((-1, *sz[-3:])), size=pre_sample_size, mode=mode,
                                     align_corners=align_corners)

        x_l_coef = new_x_l_coef.view((sz[0], sz[1], *new_x_l_coef.shape[1:]))
        x_u_coef = new_x_u_coef.view((sz[0], sz[1], *new_x_u_coef.shape[1:]))

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_batch_norm(self, current_mean: Tensor, current_var: Tensor, weight: Tensor, bias: Tensor, eps: Optional[float]=1e-5) -> "DeepPoly":
        c = (weight / torch.sqrt(current_var + eps))
        b = -current_mean * c + (0 if bias is None else bias)
        view_dim = (1, 1, -1) + (self.x_l_coef.dim()-3)*(1,)

        if self.x_l_coef.dim() == 3: #1d
            x_l_bias = self.x_l_bias + self.x_l_coef.matmul(b)
            x_u_bias = self.x_u_bias + self.x_u_coef.matmul(b)
        elif self.x_l_coef.dim() == 5: #2d
            x_l_bias = self.x_l_bias + (self.x_l_coef*b.view(view_dim)).sum((-1,-2,-3))
            x_u_bias = self.x_u_bias + (self.x_u_coef*b.view(view_dim)).sum((-1,-2,-3))
        else:
            raise NotImplementedError

        x_l_coef = self.x_l_coef*c.view(view_dim)
        x_u_coef = self.x_u_coef*c.view(view_dim)

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_group_norm(self, num_groups:int,weight: Tensor, bias: Tensor,bounds: Tuple[Tensor], eps: Optional[float]=1e-5) -> "DeepPoly":
        lb, ub = bounds
        B,C,H,W= lb.shape
        lb = lb.view(B,num_groups,-1)
        ub = ub.view(B,num_groups,-1)

        exp_l,exp_u = lb.mean(2), ub.mean(2)

        temp_l, temp_u = lb.clone() - exp_u[:,:,None].clone(), ub.clone() - exp_l[:,:,None].clone()
        #square temp_l, temp_u
        var_l, var_u = torch.zeros_like(temp_l),torch.zeros_like(temp_u)

        var_l = torch.where((temp_l >= 0.0), temp_l.pow(2),var_l)
        var_u = torch.where((temp_l >= 0.0), temp_u.pow(2),var_u)

        var_l = torch.where((temp_u <= 0.0),temp_u.pow(2),var_l)
        var_u = torch.where((temp_u <= 0.0), temp_l.pow(2),var_u)

        std_l = (var_l.mean(2) + eps).sqrt()
        std_u = (var_u.mean(2) + eps).sqrt()

        if C // num_groups != 1:
            t1 = torch.zeros((B,C)).type(lb.type()).to(lb.device)
            t2 = torch.zeros((B,C)).type(lb.type()).to(lb.device)
            t3 = torch.zeros((B,C)).type(lb.type()).to(lb.device)
            t4 = torch.zeros((B,C)).type(lb.type()).to(lb.device)
            for i in range(num_groups):
                t1[:,2*i] = exp_l[:,i]
                t1[:,2*i + 1] = exp_l[:,i]

                t2[:,2*i] = exp_u[:,i]
                t2[:,2*i + 1] = exp_u[:,i]

                t3[:,2*i] = std_l[:,i]
                t3[:,2*i + 1] = std_l[:,i]

                t4[:,2*i] = std_u[:,i]
                t4[:,2*i + 1] = std_u[:,i]
            exp_l = t1
            exp_u = t2
            std_l = t3
            std_u = t4

        lb = lb.view(B,C,H,W)
        ub = ub.view(B,C,H,W)

        temp_l = temp_l.view(B,C,H,W)
        temp_u = temp_u.view(B,C,H,W)

        z_u = torch.max((weight/ std_l),(weight / std_u))
        z_l = torch.min((weight/ std_l),(weight / std_u))

        lambda_u = torch.zeros_like(lb)
        lambda_l = torch.zeros_like(lb)

        lambda_u = torch.where(temp_l >= 0.0, z_u.view(B, C, 1, 1), lambda_u)
        lambda_l = torch.where(temp_l >= 0.0, z_l.view(B, C, 1, 1), lambda_l)

        lambda_u = torch.where(temp_u <= 0.0, z_l.view(B, C, 1, 1), lambda_u)
        lambda_l = torch.where(temp_u <= 0.0, z_u.view(B, C, 1, 1), lambda_l)

        w_std_l = (weight/ std_l)
        w_std_u = (weight/ std_u)

        bias_u_crossing = torch.max(w_std_l.view(B, C, 1, 1) * temp_u, w_std_l.view(B, C, 1, 1)* temp_l)
        bias_l_crossing = torch.min(w_std_l.view(B, C, 1, 1) * temp_u, w_std_l.view(B, C, 1, 1)* temp_l)

        mu_u = bias.view(B, C, 1, 1) + torch.where((temp_u > 0.0) * (temp_l < 0.0), bias_u_crossing, torch.zeros_like(bias_u_crossing))
        mu_l = bias.view(B, C, 1, 1) + torch.where((temp_u > 0.0) * (temp_l < 0.0), bias_l_crossing, torch.zeros_like(bias_l_crossing))

        temp1 =  (-exp_l).view(B, C, 1, 1) * (torch.where(temp_l >= 0.0, w_std_l.view(B, C, 1, 1), torch.zeros_like(temp_l)) + torch.where(temp_u <= 0.0, w_std_u.view(B, C, 1, 1), torch.zeros_like(temp_l)))
        temp2 =  (-exp_u).view(B, C, 1, 1) * (torch.where(temp_l >= 0.0, w_std_u.view(B, C, 1, 1), torch.zeros_like(temp_l)) + torch.where(temp_u <= 0.0, w_std_l.view(B, C, 1, 1), torch.zeros_like(temp_l)))

        mu_u = mu_u + torch.where(weight.view(B,C,1,1)>=0.0,temp1,temp2)
        mu_l = mu_l + torch.where(weight.view(B,C,1,1)<0.0,temp1,temp2)

        lambda_l, lambda_u = lambda_l.unsqueeze(1), lambda_u.unsqueeze(1)
        mu_l, mu_u = mu_l.unsqueeze(1), mu_u.unsqueeze(1)

        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)

        x_l_coef = pos_x_l_coef * lambda_l + neg_x_l_coef * lambda_u
        new_x_l_bias = pos_x_l_coef * mu_l + neg_x_l_coef * mu_u
        x_u_coef = pos_x_u_coef * lambda_u + neg_x_u_coef * lambda_l
        new_x_u_bias = pos_x_u_coef * mu_u + neg_x_u_coef * mu_l

        if len(new_x_l_bias.size()) == 3:
            new_x_l_bias = new_x_l_bias.sum(2)
            new_x_u_bias = new_x_u_bias.sum(2)
        else:
            new_x_l_bias = new_x_l_bias.sum((2, 3, 4))
            new_x_u_bias = new_x_u_bias.sum((2, 3, 4))

        x_l_bias = self.x_l_bias + new_x_l_bias
        x_u_bias = self.x_u_bias + new_x_u_bias

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_res_block(self, residual, downsample, relu_final, it, dp_lambda):
        in_dp_elem = self

        if relu_final is not None:
            in_dp_elem = in_dp_elem.dp_relu(relu_final.bounds, it, dp_lambda["relu_final"] if dp_lambda is not None and relu_final in dp_lambda else None)

        id_dp_elem = DeepPoly(in_dp_elem.x_l_coef, in_dp_elem.x_u_coef)

        res_dp_elem = backprop_dp(residual, in_dp_elem, it, dp_lambda["residual"] if dp_lambda is not None and "residual" in dp_lambda else None)

        if downsample is not None:
            id_dp_elem = backprop_dp(downsample, id_dp_elem, it, dp_lambda["downsample"] if dp_lambda is not None and "downsample" in dp_lambda else None)

        out_dp_elem = id_dp_elem.dp_add(res_dp_elem)

        return out_dp_elem

    def dp_partial_concretize(self,data):
        #assumes that we concretize the last x dimensions
        #assumes data is HybridZonotope
        start_dim = self.x_l_coef.shape[-1]- data.shape[-1]
        temp = DeepPoly(self.x_l_coef[:,:,start_dim:], self.x_u_coef[:,:,start_dim:])
        bounds = data.concretize()
        bias_l, bias_u = temp.dp_concretize(bounds)
        return DeepPoly(self.x_l_coef[:,:,:start_dim],self.x_u_coef[:,:,:start_dim], self.x_l_bias + bias_l, self.x_u_bias + bias_u)

    def dp_mixed_identity_abs(self,bound):
        start_dim = self.x_l_coef.shape[-1] // 2
        #abs(x) = max(x,-x) = -x + max(0,2x) = -x + ReLU(2x) = -x + 2*ReLU(x) 

        temp = DeepPoly(self.x_l_coef[:,:,start_dim:],self.x_u_coef[:,:,start_dim:])

        absolute = temp.clone().dp_relu(bound, 0, None).dp_linear(weight = torch.tensor(2).to(self.device).type(bound[0].dtype), bias = None)
        negative = temp.dp_linear(weight = torch.tensor(-1).to(self.device).type(bound[0].dtype),bias = None)
        absolute = absolute.dp_add(negative)

        new_x_l_coef = torch.cat((self.x_l_coef[:,:,0:start_dim],absolute.x_l_coef),-1)
        new_x_u_coef = torch.cat((self.x_u_coef[:,:,0:start_dim],absolute.x_u_coef),-1)

        
        new_x_l_bias = self.x_l_bias + absolute.x_l_bias
        new_x_u_bias = self.x_u_bias + absolute.x_u_bias

        return DeepPoly(new_x_l_coef,new_x_u_coef, new_x_l_bias, new_x_u_bias)
    def dp_multiplication_with_interval(self,interval,own_bounds):
        #get operation region of own state
        own_lb,own_ub = own_bounds[0],own_bounds[1]
        int_lb,int_ub = interval[0],interval[1]

        is_positive = torch.where((own_lb> 0))
        is_negative = torch.where((own_ub<=0))
        is_crossing = torch.where((own_lb<= 0) * (own_ub>0))

        if is_crossing[0].shape[0] > 0:
            t1, t2,t3,t4 = int_lb[is_crossing] * own_lb[is_crossing] ,int_ub[is_crossing] * own_lb[is_crossing], int_lb[is_crossing] * own_ub[is_crossing] ,int_ub[is_crossing] * own_ub[is_crossing]
            c_max = torch.maximum(t1,torch.maximum(t2,torch.maximum(t3,t4)))
            c_min = torch.minimum(t1,torch.minimum(t2,torch.minimum(t3,t4)))

        assert (is_positive[0].shape[0] + is_negative[0].shape[0] + is_crossing[0].shape[0]) == own_bounds[0].shape[-1]

        new_x_l_coef = torch.zeros_like(self.x_l_coef)
        new_x_u_coef = torch.zeros_like(self.x_u_coef)

        new_x_l_bias = self.x_l_bias.clone() if len(self.x_l_bias.shape) >1 else torch.zeros(self.x_l_coef.shape[0:2]).to(self.x_l_coef.device).type(self.x_l_coef.dtype)
        new_x_u_bias = self.x_u_bias.clone() if len(self.x_u_bias.shape) >1 else torch.zeros(self.x_u_coef.shape[0:2]).to(self.x_u_coef.device).type(self.x_u_coef.dtype)

        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)


        for ii in range(self.x_l_coef.shape[1]): 
            if is_crossing[0].shape[0] > 0:
                new_x_u_bias[:,ii] += (c_max * pos_x_u_coef[:,ii][is_crossing] + c_min *  neg_x_u_coef[:,ii][is_crossing]).sum()
                new_x_l_bias[:,ii] += (c_min * pos_x_l_coef[:,ii][is_crossing] + c_max *  neg_x_l_coef[:,ii][is_crossing]).sum()

            new_x_u_coef[:,ii][is_positive] = int_ub[is_positive] * pos_x_u_coef[:,ii][is_positive] + int_lb[is_positive] * neg_x_u_coef[:,ii][is_positive] 
            new_x_u_coef[:,ii][is_negative] = int_lb[is_negative] * pos_x_u_coef[:,ii][is_negative] + int_ub[is_negative] * neg_x_u_coef[:,ii][is_negative] 

            new_x_l_coef[:,ii][is_positive] = int_lb[is_positive] * pos_x_l_coef[:,ii][is_positive] + int_ub[is_positive] * neg_x_l_coef[:,ii][is_positive] 
            new_x_l_coef[:,ii][is_negative] = int_ub[is_negative] * pos_x_l_coef[:,ii][is_negative] + int_lb[is_negative] * neg_x_l_coef[:,ii][is_negative] 


        return DeepPoly(new_x_l_coef,new_x_u_coef, new_x_l_bias,new_x_u_bias)

        


    def dp_GRU(self, data,update, reset, new_state, mask, bounds,it):

        y_mean_options = []
        y_state_options = []
        y = self

        if mask.sum() == 0:
            return y
        y_no_bias= DeepPoly(y.x_l_coef, y.x_u_coef)


        #go back (- update + 1) * g_box
        dp_new_state = y.clone().dp_multiplication_with_interval(bounds['state_box'],bounds['inverse_update_box'])
        dp_new_state = dp_new_state.dp_linear(weight = torch.tensor(-1).to(self.device).type(data.dtype), bias = torch.ones(dp_new_state.x_l_coef.shape[-1]).to(self.device).type(data.dtype))
        special_treatment = [len(update)-1, len(update)-2]
        dp_new_state = backprop_dp_with_bounds(update, dp_new_state, it= it, use_lambda=False, bounds = bounds["update_bounds"].copy(),special_treatment = special_treatment)
        dp_new_state = dp_new_state.dp_partial_concretize(data) 
        y_state_options.append(dp_new_state)

        #go back g * (-update + 1)
        dp_new_state = y.clone().dp_multiplication_with_interval(bounds['inverse_update_box'],bounds['state_box'])
        dp_new_state = dp_new_state.dp_mixed_identity_abs(bounds["state_abs_box"])
        dp_new_state = backprop_dp_with_bounds(new_state, dp_new_state, it= it, use_lambda=False, bounds = bounds["state_bounds"])
        dp_new_state = dp_new_state.dp_partial_concretize(data) 
        
        #option A, go back reset * y_box
        dp_A= dp_new_state.dp_multiplication_with_interval(bounds['y_box'],bounds['reset_box'])
        special_treatment = [len(reset)-1, len(reset)-2]
        dp_A = backprop_dp_with_bounds(reset, dp_A , it= it, use_lambda=False, bounds = bounds["reset_bounds"],special_treatment = special_treatment)
        dp_A = dp_A.dp_partial_concretize(data)
        y_state_options.append(dp_A)

        #option B go back y_box * reset_box
        dp_new_state = dp_new_state.dp_multiplication_with_interval(bounds['reset_box'],bounds['y_box'])
        y_state_options.append(dp_new_state)

        # update * y_box
        special_treatment = [len(update)-1, len(update)-2]
        dp_update = y_no_bias.dp_multiplication_with_interval(bounds['y_box'],bounds['update_box'])
        dp_update = backprop_dp_with_bounds(update, dp_update, it= it, use_lambda=False, bounds = bounds["update_bounds"],special_treatment = special_treatment)
        dp_update = dp_update.dp_partial_concretize(data) 
        y_mean_options.append(dp_update)

        # update_box * y
        dp_y_mean = y_no_bias.dp_multiplication_with_interval(bounds['update_box'],bounds['y_box'])
        y_mean_options.append(dp_y_mean)

        y_sols = []
        for y1 in y_state_options:
            for y2 in y_mean_options:
                y_sols.append(y1.clone().dp_add(y2))

        out = y_sols.pop(0)
        
        l1,u1 = out.dp_concretize(bounds['y_box'])
        out_x_l_coef = out.x_l_coef
        out_x_u_coef = out.x_u_coef
        out_x_l_bias = out.x_l_bias
        out_x_u_bias = out.x_u_bias
        #tightest bounds method
        
        for y in y_sols:
            l2,u2 = y.dp_concretize(bounds['y_box'])

            # take smallest ub per entry
            upper = torch.where(u2 < u1)
            out_x_u_coef[upper] = y.x_u_coef[upper]
            out_x_u_bias[upper] = y.x_u_bias[upper]
            u1[upper] = u2[upper]

            #smallest lb coefs
            lower= torch.where(l2 > l1)
            out_x_l_coef[lower] = y.x_l_coef[lower]
            out_x_l_bias[lower] = y.x_l_bias[lower]
            l1[lower] = l2[lower]
        '''
        
        #most relations method
        numb_low_relations = (out_x_l_coef != 0.).int().sum(2)
        numb_upp_relations = (out_x_u_coef != 0.).int().sum(2)

        for y in y_sols:
            l2,u2 = y.dp_concretize(bounds['y_box'])
            temp_numb_low = (y.x_l_coef  != 0.).int().sum(2)
            temp_numb_upp = (y.x_u_coef  != 0.).int().sum(2)

            upper = torch.where(temp_numb_upp > numb_upp_relations)
            out_x_u_coef[upper] = y.x_u_coef[upper]
            out_x_u_bias[upper] = y.x_u_bias[upper]
            u1[upper] = u2[upper]
            numb_upp_relations[upper] = temp_numb_upp[upper]

            upper = torch.where((temp_numb_upp == numb_upp_relations) * (u2 < u1))
            out_x_u_coef[upper] = y.x_u_coef[upper]
            out_x_u_bias[upper] = y.x_u_bias[upper]
            u1[upper] = u2[upper]
            numb_upp_relations[upper] = temp_numb_upp[upper]

            lower= torch.where(temp_numb_low > numb_low_relations)
            out_x_l_coef[lower] = y.x_l_coef[lower]
            out_x_l_bias[lower] = y.x_l_bias[lower]
            l1[lower] = l2[lower]
            numb_low_relations[lower] = temp_numb_low[lower]

            lower= torch.where((temp_numb_low == numb_low_relations) * (l2 > l1))
            out_x_l_coef[lower] = y.x_l_coef[lower]
            out_x_l_bias[lower] = y.x_l_bias[lower]
            l1[lower] = l2[lower]
            numb_low_relations[lower] = temp_numb_low[lower]
        
        '''
        return DeepPoly(out_x_l_coef,out_x_u_coef,out_x_l_bias,out_x_u_bias)


    def dp_ODEBlock(self, ODEBlock, method: str,integration_time: Union[Tensor,float],input_bounds: Tuple[Tensor],graph = None, running_mean_step = None,mode:int = 0,deepz_lambda_dict = {},it = 0):
        if method == "rk4_10":
            met = "rk4"
            n_steps = 10
            dt = integration_time / 10
        elif method == "rk4_0.1":

            met = "rk4"
            dt = 0.1
            n_steps = int(integration_time / 0.1)
        elif method == "rk4_1":
            met = "rk4"
            dt = integration_time
            n_steps = 1
        elif method == "euler":
            met = "euler"
            dt = integration_time
            n_steps = 1
        elif method == "euler_2":
            met = "euler"
            dt = integration_time /2
            n_steps = 2
        elif method == "mp_2":
            met = "midpoint"
            dt = integration_time
            n_steps = 1
        elif method in ["dopri5_0.005_2a","dopri5_0.01_2a","dopri5_0.001_2a","dopri5_0.0001_2a","bosh3_0.005_2a"]:
            dt = ODEBlock.running_mean_step
        else:
            print("Not implemented method:",method)
            print(aga)
            exit() 

        self.use_lambda = False

        if len(deepz_lambda_dict.keys()) > 0:
            self.use_lambda = True

        t = 0.0
        out = self.clone()
        bounds = [temp.clone() for temp in input_bounds]

        if method in ["rk4_1","dopri5_0.005_2a","dopri5_0.01_2a","dopri5_0.001_2a","dopri5_0.0001_2a","bosh3_0.005_2a","euler_2"]:
            if method in ["rk4_1"]:
                running_mean_step = 1
            if method in ["euler_2"]:
                running_mean_step = integration_time / 2

            graph,prev_bounds = graph
            max_key = max(graph.keys())
            temporary_graph = {max(graph.keys()):[out.clone()]}
            similarities = []

            d_l,c_l,a_l = [],[],[]

            graph_keys = list(graph.keys())

            while len(graph.keys())>0:
                if len(graph_keys )== 0:
                    break

                key1 = max(graph_keys)
                graph_keys.remove(key1)


                y_list = temporary_graph.pop(key1)
                for key2 in list(graph[key1].keys()):
                    _bounds = graph[key1][key2]
                    bounds = _bounds[0][0]

                    bounds_list = _bounds[0][1]
                    #constraint aggregation for each edge individually
                    if len(y_list) == 1:
                        y1 = y_list[0]
                    else:
                        y1,sim  = relu_constraint_aggregation(y_list,bounds,True)
                        if len(sim) >0:
                            similarities.append(np.mean(sim))

                    t0 = key2[0] * running_mean_step * (1  - key2[2]) + integration_time * key2[2]
                    dt = -(key2[1]) * running_mean_step  - key2[3] * (integration_time - key2[0] * running_mean_step)

                    if len(bounds_list) == 0: #not accepted step
                        y0 = y1
                    else:
                        if method == "rk4_1":
                            y0 = y1.clone().dp_rk4_step_b(ODEBlock,t0,dt,bounds_list)
                        elif method == "euler_2":
                            y0 = y1.clone().dp_euler_step_b(ODEBlock,t0,dt,bounds_list)
                        elif method in ["bosh3_0.005_2a"]:
                            y0 = y1.clone().dp_bosh3_step(ODEBlock,t0,dt,bounds_list)
                        elif method in ["dopri5_0.005_2a","dopri5_0.01_2a","dopri5_0.001_2a","dopri5_0.0001_2a"]:

                            zz = y1.clone()
                            if self.use_lambda:
                                zz.use_lambda = self.use_lambda
                                zz.key = list(key2)
                                zz.deepz_lambda_dict = deepz_lambda_dict
                            else:
                                zz.use_lambda = False
                            y0,dead,cross,active = zz.dp_dopri5_step(ODEBlock,t0,dt,bounds_list,it,True)
                            d_l.append(dead)
                            c_l.append(cross)
                            a_l.append(active)

                        else:
                            print("alarm")
                            exit()
                    if key2 in temporary_graph.keys():
                        temporary_graph[key2].append(y0) 
                    else:
                        temporary_graph[key2] = [y0]

            assert len(temporary_graph.keys()) ==1, "graph finished but multiple endpoints issue"
            key = max(temporary_graph.keys())
            y0 = temporary_graph.pop(key)

            if len(y0) == 1:
                return y0[0],similarities,np.mean(d_l),np.mean(c_l),np.mean(a_l)
            else:
                y1,sim  = relu_constraint_aggregation(y0,prev_bounds,True)
                if len(sim) >0:
                    similarities.append(np.mean(sim))
                return y1, similarities,np.mean(d_l),np.mean(c_l),np.mean(a_l)

        elif met in ["rk4","euler","midpoint"]:
            if mode == 0: 
                for i in range(n_steps):
                    if met =="rk4":
                        out = out.dp_rk4_step(ODEBlock,t,dt,bounds)
                    elif met == "euler":
                        out = out.dp_euler_step(ODEBlock,t,dt,bounds)
                    elif met == "midpoint":
                        out = out.dp_midpoint_step(ODEBlock,t,dt,bounds)
                    else:
                        print("fuck youu")

                    t += dt
                    temp_bounds = [temp.clone() for temp in input_bounds]
                    bounds = out.dp_concretize(bounds = temp_bounds)
                    bounds = (bounds[0].view(*temp_bounds[0].shape),bounds[1].view(*temp_bounds[1].shape))
            else:
                if met == "euler":
                    out = out.dp_full_euler(ODEBlock,n_steps,dt,bounds)

                else: 
                    print("f**k you")
                    print(aga)
            ODEBlock.forward_computed = True
        return out


    def dp_rk4_step_b(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds):
        #returns output of backward RK4 iteration, inequalities expressed in the terms/level of the input variable x
        alpha = torch.tensor([0, 1 / 3, 2 / 3, 1.] * dt, dtype=torch.float64).to(self.device)
        beta = [torch.tensor([1 / 3], dtype=torch.float64).to(self.device) * dt,
                torch.tensor([-1 / 3, 1.], dtype=torch.float64).to(self.device)* dt,
                torch.tensor([1., -1., 1.], dtype=torch.float64).to(self.device) * dt,
                ]
        c_sol =torch.tensor([1/8, 3/8, 3 / 8, 1 / 8], dtype=torch.float64).to(self.device) * dt

        filler_elem = DeepPoly(self.x_l_coef, self.x_u_coef)
        liste = [self.clone(),filler_elem.dp_linear(c_sol[0],None), filler_elem.dp_linear(c_sol[1],None), 
                    filler_elem.dp_linear(c_sol[2],None), filler_elem.dp_linear(c_sol[3],None)]

        for i in range(len(liste)-1,0,-1):
            k_dp = liste.pop(i).dp_odefunc_b(ODEBlock,t0 + alpha[i-1], bounds["k{0}".format(i)])
            liste[0] = liste[0].dp_add(k_dp)
            filler_elem = DeepPoly(k_dp.x_l_coef, k_dp.x_u_coef)
            for j in range(1,len(liste)):
                liste[j] = liste[j].dp_add(filler_elem.dp_linear(beta[i-2][j-1],None))
        return liste[0]

    def dp_euler_step_b(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds):
        #returns output of backward RK4 iteration, inequalities expressed in the terms/level of the input variable x
        alpha = torch.tensor([0], dtype=torch.float32).to(self.device)
        c_sol =torch.tensor([1], dtype=torch.float32).to(self.device) * dt
        filler_elem = DeepPoly(self.x_l_coef, self.x_u_coef)
        liste = [self.clone(),filler_elem.dp_linear(c_sol[0],None)]
        for i in range(len(liste)-1,0,-1):
            k_dp = liste.pop(i).dp_odefunc_b(ODEBlock,t0 + alpha[i-1], bounds["k{0}".format(i)])
            liste[0] = liste[0].dp_add(k_dp)
            filler_elem = DeepPoly(k_dp.x_l_coef, k_dp.x_u_coef)
            for j in range(1,len(liste)):
                liste[j] = liste[j].dp_add(filler_elem.dp_linear(beta[i-2][j-1],None))
        return liste[0]
     
    def dp_bosh3_step(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds):
        #returns output of backward bosh3 iteration, inequalities expressed in the terms/level of the input variable x
        alpha = torch.tensor([0,1 / 2, 3 / 4, 1.], dtype=torch.float64).to(self.device) * dt #ADDITIONAL 0

        beta = [torch.tensor([1 / 2], dtype=torch.float64).to(self.device) * dt,
                torch.tensor([0., 3 / 4], dtype=torch.float64).to(self.device) * dt,
                torch.tensor([2 / 9, 1 / 3, 4 / 9], dtype=torch.float64).to(self.device) * dt]
        c_sol = torch.tensor([2 / 9, 1/3, 4 / 9, 0], dtype=torch.float64).to(self.device) * dt


        filler_elem = DeepPoly(self.x_l_coef, self.x_u_coef)
        liste = [self.clone(),filler_elem.dp_linear(c_sol[0],None), filler_elem.dp_linear(c_sol[1],None), 
                    filler_elem.dp_linear(c_sol[2],None), filler_elem.dp_linear(c_sol[3],None)]

        for i in range(len(liste)-1,0,-1):
            k_dp = liste.pop(i).dp_odefunc_b(ODEBlock,t0 + alpha[i-1], bounds["k{0}".format(i)].copy())
            liste[0] = liste[0].dp_add(k_dp)
            filler_elem = DeepPoly(k_dp.x_l_coef, k_dp.x_u_coef)
            for j in range(1,len(liste)):
                liste[j] = liste[j].dp_add(filler_elem.dp_linear(beta[i-2][j-1],None))
        return liste[0]

    def dp_dopri5_step(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds,it,mode = False):
        #returns output of backward dopri5 iteration, inequalities expressed in the terms/level of the input variable x
        alpha = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.], dtype=torch.float64).to(self.device) * dt

        beta = [torch.tensor([1 / 5], dtype=torch.float64).to(self.device) * dt,
                torch.tensor([3 / 40, 9 / 40], dtype=torch.float64).to(self.device) * dt,
                torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=torch.float64).to(self.device) * dt,
                torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=torch.float64).to(self.device) * dt,
                torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=torch.float64).to(self.device) * dt]

        c_sol =torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=torch.float64).to(self.device) * dt


        filler_elem = DeepPoly(self.x_l_coef, self.x_u_coef)
        liste = [self.clone(),filler_elem.dp_linear(c_sol[0],None), filler_elem.dp_linear(c_sol[1],None), 
                    filler_elem.dp_linear(c_sol[2],None), filler_elem.dp_linear(c_sol[3],None),
                    filler_elem.dp_linear(c_sol[4],None), filler_elem.dp_linear(c_sol[5],None)]
        if mode:
            d_l = []
            c_l = []
            a_l = []

        for i in range(len(liste)-1,0,-1):
            if self.use_lambda:
                temp_count = 0
                temp_key = self.key.copy()
                temp_key.append("k{0}".format(i))

                for ii in range(len(ODEBlock.odefunc.layers)):
                    if isinstance(ODEBlock.odefunc.layers[ii], ReLU):
                        temp_keys = temp_key.copy()
                        temp_keys.append(temp_count)
                        temp_keys = tuple(temp_keys)
                        
                        ODEBlock.odefunc.layers[ii].deepz_lambda = self.deepz_lambda_dict[temp_keys]
                        temp_count += 1
                        ODEBlock.to(self.device)
            if mode:
                k_dp,dead,cross,active = liste.pop(i).dp_odefunc_b(ODEBlock,t0 + alpha[i-1],it, bounds["k{0}".format(i)].copy(),mode = mode, use_lambda = self.use_lambda)
                d_l.append(dead.cpu().numpy())
                c_l.append(cross.cpu().numpy())
                a_l.append(active.cpu().numpy())
            else:

                k_dp = liste.pop(i).dp_odefunc_b(ODEBlock,t0 + alpha[i-1],it, bounds["k{0}".format(i)].copy(),mode = mode,use_lambda = self.use_lambda)

            liste[0] = liste[0].dp_add(k_dp)
            filler_elem = DeepPoly(k_dp.x_l_coef, k_dp.x_u_coef)
            for j in range(1,len(liste)):
                liste[j] = liste[j].dp_add(filler_elem.dp_linear(beta[i-2][j-1],None))
        if mode: 
            return liste[0], np.mean(d_l)*100,np.mean(c_l)* 100,np.mean(a_l)*100
        return liste[0]

    def dp_odefunc_b(self,ODEBlock,t: Union[Tensor,float], it,bounds,mode = False,use_lambda = False):
        # given input t and bounds at needed places, do backprob of odefunc eval
        #set bounds for layers which need them
        count1 = 0
        count2 = 0
        count3 = 0
        count = 0
        for i in range(len(ODEBlock.odefunc.layers)):
            if ODEBlock.odefunc.layers[i].needs_bounds:
                ODEBlock.odefunc.layers[i].update_bounds(bounds.pop(0))
        dp = self.clone()
        for i in range(len(ODEBlock.odefunc.layers)-1,-1,-1):
            if isinstance(ODEBlock.odefunc.layers[i],ReLU) and mode:
                count += 1
                abc = ODEBlock.odefunc.layers[i].bounds
                temp = (abc[0]<0) * (abc[1]>0)
                count1 += temp.float().mean()
                count2 += (abc[0] >= 0).float().mean()
                count3 += (abc[1]<= 0).float().mean()
            dp = backprop_dp(ODEBlock.odefunc.layers[i], dp, it= it, use_lambda=use_lambda,time = t)
        if mode:
            return dp, count3/count,count1/count,count2/count
        return dp

    def dp_rk4_step(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds: Tuple[Tensor]):
        #returns output of forward RK4 iteration, inequalities expressed in the terms/level of the input variable x
        x = self.clone()
        k1 = x.dp_odefunc(ODEBlock,t0,dt,"k1_rk4",bounds)#k1 = f(t0,y0), returns k1 *[coefs] list
        k2 = x.dp_odefunc(ODEBlock,t0 + (dt/3),dt,"k2_rk4",bounds,[k1[0]]) #k2 = f(t0 + 1/3*dt,y0 + 1/3*dt*k1) returns k2 *[coefs] list
        k3 = x.clone().dp_odefunc(ODEBlock,t0 + (dt*2/3),dt,"k3_rk4",bounds,[k1[1],k2[0]])#k3 = f(t0 + 2/3*dt,y0  -1/3*k1*dt + k2*dt) returns k3 *[coefs] list
        k4 = x.clone().dp_odefunc(ODEBlock,t0 + dt, dt, "k4_rk4",bounds,[k1[2],k2[1],k3[0]]) #k4 = f(t0 + 1*dt,y0 +dt*(k1-k2+k3)) returns k4 *[coefs] list
        
        final = x.clone().dp_add(k1[3])
        final = final.dp_add(k2[2])
        final = final.dp_add(k3[1])
        final = final.dp_add(k4[0])
        return final

    def dp_odefunc(self,ODEBlock,t: Union[Tensor,float], dt: Union[Tensor,float], kterm: str, bounds: Tuple[Tensor], add_input =()):
        # given current time t and abstract Element we want to evaluate k1,k2,k3....

        #ASSUME input bounds are for self DeepPoly
        input_low,input_upp = bounds

        inputs = self.clone()
        for i in range(len(add_input)):
            inputs = inputs.dp_add(add_input[i])

        #obtain needed coefs to precompute all needed versions of kterms
        if kterm == "k1_rk4":
            l = [1/3,-1/3,1,1/8]
            coef = [dt * temp for temp in l]
        elif kterm == "k2_rk4":
            l = [1,-1,3/8]
            coef = [dt * temp for temp in l]
        elif kterm == "k3_rk4":
            l = [1,3/8]
            coef = [dt * temp for temp in l]
        elif kterm == "k4_rk4":
            coef = [dt * 1/8]
        elif kterm == "k1_euler":
            coef = [dt]
        elif kterm == "k1_midpoint":
            coef = [dt/2]
        elif kterm == "k2_midpoint":
            coef = [dt]
        else:
            print("Unknown combination of method and kterm {0}".format(kterm))
            exit()

        #get bounds for layers which need them
        idx = []
        for i in range(len(ODEBlock.odefunc.layers)):
            if ODEBlock.odefunc.layers[i].needs_bounds:
                idx.append(i)

        for j in idx:
            temp = 1
            for y in ODEBlock.odefunc.layers[j-1].output_dim:
                temp *= y
            expr_coef = torch.eye(temp).view(-1, *ODEBlock.odefunc.layers[j-1].output_dim).unsqueeze(0).type(torch.DoubleTensor)
            dp = DeepPoly(expr_coef = expr_coef)

            for i in range(j-1,-1,-1):
                dp = backprop_dp(ODEBlock.odefunc.layers[i], dp, it= 10, use_lambda=False,time = t)
            #correct residual-like addition
            dp = dp.dp_backsub(inputs) 
            ODEBlock.odefunc.layers[j].update_bounds(dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone())))

        k = []
        temp = 1
        for y in ODEBlock.odefunc.layers[-1].output_dim:
            temp *= y
        expr_coef = torch.eye(temp).view(-1, *ODEBlock.odefunc.layers[-1].output_dim).unsqueeze(0).type(torch.DoubleTensor)

        for c in coef:
            dp = DeepPoly(expr_coef = c* expr_coef.clone())
            for i in range(len(ODEBlock.odefunc.layers)-1,-1,-1):
                dp = backprop_dp(ODEBlock.odefunc.layers[i], dp, it= 10, use_lambda=False,time = t)
            dp = dp.dp_backsub(inputs) 
            k.append(dp.clone())
        return k

    def dp_euler_step(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds: Tuple[Tensor]):
        x = self.clone()
        k1 = x.dp_odefunc(ODEBlock,t0,dt,"k1_euler",bounds)
        final = x.clone().dp_add(k1[0])
        return final

    def dp_euler_step_with_bounds(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds,it):
        #multiply by dt and run back with bounds
        x_no_bias = DeepPoly(self.x_l_coef * dt, self.x_u_coef * dt)
        k1 = backprop_dp_with_bounds(ODEBlock,x_no_bias,it= it, use_lambda=False, bounds = bounds)
        final = self.dp_add(k1)
        return final

    def dp_midpoint_step(self,ODEBlock,t0: Union[Tensor,float],dt: Union[Tensor,float],bounds: Tuple[Tensor]):
        x = self.clone()
        k1 = x.dp_odefunc(ODEBlock,t0,dt,"k1_midpoint",bounds)
        k2 = x.dp_odefunc(ODEBlock,t0 + (dt/2),dt,"k2_midpoint",bounds,[k1[0]]) 
        final = x.clone().dp_add(k2[0])
        return final

    def dp_full_euler_real_one(self,ODEBlock,steps: Union[Tensor,int], dt: Union[Tensor,float],bounds: Tuple[Tensor]):

        t = torch.zeros(1).type(dt.type())
        temp_model = []
        idx = []
        times_idx = []
        times = []
        for i in range(steps):
            temp_model.append("residual")

            for j in range(len(ODEBlock.odefunc.layers)):
                temp_model.append(ODEBlock.odefunc.layers[j])
                if ODEBlock.odefunc.layers[j].needs_bounds:
                    idx.append(j + 1+ (len(ODEBlock.odefunc.layers)+1)*i)

            for j in ODEBlock.requires_time:
                times_idx.append(j + 1 + (len(ODEBlock.odefunc.layers)+1)*i)
                times.append(t)
            t = t.clone() + dt.clone()


        input_low,input_upp = bounds

        saved_bounds = []
        length = len(ODEBlock.odefunc.layers) + 1
        idx.append(len(temp_model))
        for j in idx:
            temp = 1

            if j == len(temp_model):
                dp = self.clone()
            else:
                for y in temp_model[j-1].output_dim:
                    temp *= y
                expr_coef = torch.eye(temp).view(-1, *temp_model[j-1].output_dim).unsqueeze(0).type(torch.DoubleTensor)
                dp = DeepPoly(expr_coef = expr_coef)

            for i in range(j-1,-1,-1):
                layer = temp_model[i]
                if i % length == length -1:
                    #f is multiplied by dt
                    dp = DeepPoly(x_l_coef = dp.x_l_coef * dt,x_u_coef = dp.x_u_coef * dt,x_l_bias = dp.x_l_bias,x_u_bias = dp.x_u_bias )

                #set correct bounds
                if i in idx:
                    layer.update_bounds(saved_bounds[idx.index(i)])
                #handle addition
                if layer == "residual":
                    if i == (j - j % length):
                        checkpoint = DeepPoly(x_l_coef=dp.x_l_coef,x_u_coef = dp.x_u_coef)
                        continue
                    if j == len(temp_model) and i == (steps -1)* length:
                        checkpoint = DeepPoly(x_l_coef=self.x_l_coef,x_u_coef = self.x_u_coef)

                    dp = dp.dp_add(checkpoint)
                    checkpoint = DeepPoly(x_l_coef=dp.x_l_coef,x_u_coef = dp.x_u_coef)
                    continue
                #get correct time if needed
                if i in times_idx:
                    dp = backprop_dp(layer, dp, it= 10, use_lambda=False,time = times[times_idx.index(i)])
                else:
                    dp = backprop_dp(layer, dp, it= 10, use_lambda=False) 

            if j == len(temp_model):
                return dp

            lb, ub = dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
            saved_bounds.append((lb,ub))
    
    def dp_full_euler(self,ODEBlock,steps: Union[Tensor,int], dt: Union[Tensor,float],bounds: Tuple[Tensor]):
        time_grid = torch.tensor([0.0,0.5,1.0]).to(dt.device).type(dt.dtype)
        time_grid = torch.tensor([0.0,0.5,0.75,1.0]).to(dt.device).type(dt.dtype)
        time_grid = torch.tensor([0.0,0.5,0.625,1.0]).to(dt.device).type(dt.dtype)
        time_grid = torch.tensor([0.0,0.5,0.875,1.0]).to(dt.device).type(dt.dtype)
        t = torch.zeros(1).type(dt.type())
        temp_model = []
        idx = []
        times_idx = []
        times = []
        for i in range(len(time_grid)-1):
            temp_model.append("residual")
            #store bounds for y_0.5
            if i == 1:
                if ((len(ODEBlock.odefunc.layers)+1)*i - 1) not in idx:
                    idx.append((len(ODEBlock.odefunc.layers)+1)*i - 1)
                idx.append((len(ODEBlock.odefunc.layers)+1)*i)

            for j in range(len(ODEBlock.odefunc.layers)):
                temp_model.append(ODEBlock.odefunc.layers[j])
                if ODEBlock.odefunc.layers[j].needs_bounds:
                    idx.append(j + 1+ (len(ODEBlock.odefunc.layers)+1)*i)

            for j in ODEBlock.requires_time:
                times_idx.append(j + 1 + (len(ODEBlock.odefunc.layers)+1)*i)
                times.append(time_grid[i])

        input_low,input_upp = bounds

        saved_bounds = []
        length = len(ODEBlock.odefunc.layers) + 1
        idx.append(len(temp_model))
        for j in idx:
            temp = 1

            if j == len(temp_model):
                dp = self.clone()
            else:
                for y in temp_model[j-1].output_dim:
                    temp *= y
                expr_coef = torch.eye(temp).view(-1, *temp_model[j-1].output_dim).unsqueeze(0).type(torch.DoubleTensor)
                dp = DeepPoly(expr_coef = expr_coef)

            for i in range(j-1,-1,-1):
                layer = temp_model[i]
                if i % length == length -1:
                    #f is multiplied by dt
                    dt = time_grid[1 + (i//length)] - time_grid[(i//length)]
                    dp = DeepPoly(x_l_coef = dp.x_l_coef * dt,x_u_coef = dp.x_u_coef * dt,x_l_bias = dp.x_l_bias,x_u_bias = dp.x_u_bias )
                #set correct bounds
                if i in idx:
                    if layer != "residual":
                        layer.update_bounds(saved_bounds[idx.index(i)])
                #handle addition
                if layer == "residual":
                    if i == (j - j % length):
                        checkpoint = DeepPoly(x_l_coef=dp.x_l_coef,x_u_coef = dp.x_u_coef)
                        continue
                    if j == len(temp_model) and i == (len(time_grid)-1 -1)* length:
                        checkpoint = DeepPoly(x_l_coef=self.x_l_coef,x_u_coef = self.x_u_coef)

                    dp = dp.dp_add(checkpoint)
                    checkpoint = DeepPoly(x_l_coef=dp.x_l_coef,x_u_coef = dp.x_u_coef)

                    if j == len(temp_model) and i == length:
                        #return y0.5 constraints together with bounds
                        return (dp,saved_bounds[idx.index(i-1)],saved_bounds[idx.index(i)])
                    continue
                #get correct time if needed
                if i in times_idx:
                    dp = backprop_dp(layer, dp, it= 10, use_lambda=False,time = times[times_idx.index(i)])
                else:
                    dp = backprop_dp(layer, dp, it= 10, use_lambda=False) 

            if j == len(temp_model):
                return dp

            lb, ub = dp.dp_concretize(bounds = (input_low.clone(),input_upp.clone()))
            saved_bounds.append((lb,ub))


def backprop_dp_with_bounds(net, dp, it= 10, use_lambda=False, bounds = None,time: Optional[Union[Tensor,float]]=None,special_treatment = [] ):
    if bounds != None:
        temp_length = len(bounds) - 1
    for i in range(len(net)-1,-1,-1):
        if i in special_treatment:
            # if needs bounds, need to stack bounds twice
            if net[i].needs_bounds:
                temp_lb,temp_ub = bounds[temp_length]
                temp_length -= 1
                net[i].update_bounds((torch.cat((temp_lb,temp_lb),dim= -1).squeeze(1),torch.cat((temp_ub,temp_ub),dim= -1).squeeze(1)),leave_dim = True)
            elif isinstance(net[i],Linear):
                weight = torch.cat((net[i].weight.data,net[i].weight.data),0)
                bias = torch.cat((net[i].bias.data,net[i].bias.data),0)
                temp_layer = torch.nn.Linear(1,1)
                temp_layer.bias.data = bias
                temp_layer.weight.data = weight
                dp = backprop_dp(temp_layer, dp, it = it, use_lambda=False,time = 0.0) 
                continue
            else:
                print("unexpected layer in special treatment", type(net[i]))
                exit()

        elif net[i].needs_bounds:
            net[i].update_bounds(bounds[temp_length])
            temp_length -= 1
        dp = backprop_dp(net[i], dp, it = it, use_lambda=False,time = 0.0)   
    return dp

def backprop_dp(layer, abs_dp_element, it, use_lambda=False, bounds: Optional[Tuple[Tensor]]=None,time: Optional[Union[Tensor,float]]=None,mode: Optional[int]=0): 
    if isinstance(layer, Sequential):
        for j in range(len(layer.layers)-1, -1, -1):
            sub_layer = layer.layers[j]
            abs_dp_element = backprop_dp(sub_layer, abs_dp_element, it, use_lambda)
    elif isinstance(layer, nn.Linear):
        abs_dp_element = abs_dp_element.dp_linear(layer.weight, layer.bias)
    elif isinstance(layer, Flatten) or str(type(layer)) == "<class 'model.Flatten'>":
        abs_dp_element = abs_dp_element.dp_flatten([1,1])
    elif isinstance(layer, Normalization):
        abs_dp_element = abs_dp_element.dp_normalize(layer.mean, layer.sigma)
    elif isinstance(layer, ReLU):
        abs_dp_element = abs_dp_element.dp_relu(layer.bounds, it, layer.deepz_lambda if use_lambda else None)
    elif isinstance(layer, Conv2d):
        abs_dp_element = abs_dp_element.dp_conv(layer.dim, layer.weight, layer.bias, layer.stride, layer.padding, layer.groups, layer.dilation)
    elif isinstance(layer, ConcatConv):
        abs_dp_element = abs_dp_element.dp_concat_conv(layer._layer.dim, layer._layer.weight, layer._layer.bias, time,layer._layer.stride, layer._layer.padding, layer._layer.groups, layer._layer.dilation)
    elif isinstance(layer, GlobalAvgPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
        abs_dp_element = abs_dp_element.dp_global_avg_pool2d(layer.input_dim)
    elif isinstance(layer, AvgPool2d):
        abs_dp_element = abs_dp_element.dp_avg_pool2d(layer.dim, layer.kernel_size, layer.stride, layer.padding)
    elif isinstance(layer, Upsample):
        abs_dp_element = abs_dp_element.dp_upsample(layer.dim[-2:], layer.mode, layer.align_corners)
    elif isinstance(layer, _BatchNorm):
        if layer.training:
            mean = layer.current_mean
            var = layer.current_var
        else:
            mean = layer.running_mean
            var = layer.running_var
        abs_dp_element = abs_dp_element.dp_batch_norm(mean, var, layer.weight, layer.bias, layer.eps)
    elif isinstance(layer, BatchNorm): #modified batchnorm with multiple BN at different times
        key= (layer.timepoints - time).abs().argmin()
        abs_dp_element = abs_dp_element.dp_batch_norm(layer.BatchNorm2d[key].running_mean,layer.BatchNorm2d[key].running_var, layer.BatchNorm2d[key].weight, layer.BatchNorm2d[key].bias, layer.BatchNorm2d[key].eps)

    elif isinstance(layer, Bias):
        abs_dp_element = abs_dp_element.dp_bias(layer.bias)
    elif isinstance(layer, GroupNorm):
        abs_dp_element = abs_dp_element.dp_group_norm(layer.num_groups,layer.weight, layer.bias,layer.bounds,layer.eps)
    elif isinstance(layer, Scale):
        abs_dp_element = abs_dp_element.dp_scale(layer.scale)
    elif isinstance(layer, ResBlock):
        abs_dp_element = abs_dp_element.dp_res_block(layer.residual, layer.downsample, layer.relu_final, it, lambda_layer)
    elif isinstance(layer, ODEBlock_A):
        if mode == 0:
            if layer.graph != None:
                abs_dp_element, layer.sim,layer.dead,layer.cross,layer.active = abs_dp_element.dp_ODEBlock(layer, layer.method,layer.integration_time,layer.bounds,layer.graph,layer.running_mean_step,mode,layer.deepz_lambda_dict,it)
            else:
                abs_dp_element,layer.sim,layer.dead,layer.cross,layer.active = abs_dp_element,1,0,0,0

        elif mode == 1:
            abs_dp_element = abs_dp_element.dp_ODEBlock(layer, layer.method,layer.integration_time,layer.bounds,mode)
        else:
            print("something is off")
    elif isinstance(layer, Sigmoid):
        abs_dp_element = abs_dp_element.dp_sigmoid(layer.bounds, it, None)
    else:
        raise RuntimeError(f'Unknown layer type: {type(layer)}')
    return abs_dp_element

def backward_deeppoly(net, layer_idx, abs_dp_element, it, use_lambda=False, use_intermediate=False, abs_inputs=None):
    x_u_bias, x_l_bias = None, None

    for j in range(layer_idx, -1, -1):
        layer = net.layers[j]
        abs_dp_element = backprop_dp(layer, abs_dp_element, it, use_lambda)

        if j == 0 or (use_intermediate and layer.bounds is not None):
            x_l_bias_tmp, x_u_bias_tmp = abs_dp_element.dp_concretize(layer.bounds if j > 0 else None, None if j > 0 else abs_inputs)
            if x_u_bias is not None:
                x_l_bias = torch.maximum(x_l_bias, x_l_bias_tmp)
                x_u_bias = torch.minimum(x_u_bias, x_u_bias_tmp)
            else:
                x_l_bias = x_l_bias_tmp
                x_u_bias = x_u_bias_tmp

    return x_l_bias, x_u_bias

def get_layer_sizes(net, x):
    layer_sizes = {}
    for i, layer in enumerate(net.blocks):
        layer_sizes[i] = x.size()
        x = layer(x)
    layer_sizes[i+1] = x.size()
    return layer_sizes

def compute_dp_relu_bounds(net, max_layer_id, abs_input, it, use_lambda=False, recompute_bounds=True, use_intermediate=False):
    x = abs_input.head
    device = x.device

    if max_layer_id == 0:
        x_l_bias, x_u_bias = abs_input.concretize()
    else:
        for i, layer in enumerate(net.layers[:max_layer_id]):
            x = layer(x)
            if isinstance(layer, ReLU):
                if layer.bounds is None or recompute_bounds:
                    compute_dp_relu_bounds(net, i, abs_input, it, use_lambda, use_intermediate=use_intermediate)

        k = int(np.prod(x[0].size()))
        expr_coef = torch.eye(k).view(-1, *x[0].size()).unsqueeze(0).to(device)

        abs_dp_element = DeepPoly(expr_coef=expr_coef)
        x_l_bias, x_u_bias = backward_deeppoly(net, max_layer_id - 1, abs_dp_element, it, use_lambda, use_intermediate, abs_input)

    net.layers[max_layer_id].update_bounds((x_l_bias, x_u_bias))

def forward_deeppoly(net, abs_input, expr_coef=None, it=0, use_lambda=False, recompute_bounds=False, use_intermediate=True):
    net.set_dim(abs_input.concretize()[0][0:1])

    x = net(abs_input.head)
    if recompute_bounds:
        compute_dp_relu_bounds(net, len(net.layers)-1, abs_input, it, use_lambda=False, use_intermediate=use_intermediate)

    if expr_coef is None:
        k = int(np.prod(x[0].size()))
        abs_dp_element = DeepPoly(expr_coef=torch.eye(k).view(-1, *x[0].size()).unsqueeze(0).to(abs_input.head.device))
    else:
        abs_dp_element = DeepPoly(expr_coef=expr_coef)

    x_l_bias, x_u_bias = backward_deeppoly(net, len(net.layers) - 1, abs_dp_element, it, use_lambda, use_intermediate,
                                           abs_input)

    if expr_coef is None:
        x_l_bias = x_l_bias.view(-1, *x.size()[1:])
        x_u_bias = x_u_bias.view(-1, *x.size()[1:])

    return x_l_bias, x_u_bias


def relu_constraint_aggregation(DP, bounds,verbose = False):
    lb_prev ,ub_prev = bounds[0].flatten(1),bounds[1].flatten(1)
    shp = DP[0].x_u_coef.shape
    DP = [DeepPoly(x.x_l_coef.flatten(2), x.x_u_coef.flatten(2), x.x_l_bias, x.x_u_bias) for x in DP]
    DP_aggregated = DP[0].clone()
    low_c_1 = 0
    low_c_2 = 0
    low_c_3 = 0
    upp_c_1 = 0
    upp_c_2 = 0
    upp_c_3 = 0
    low_c_4 = 0
    upp_c_4 = 0
    similarities = []

    for i in range(DP[0].x_l_coef.shape[1]):
        constraint_low = []
        constraint_upp = []

        concrete_bounds_low = []
        concrete_bounds_upp = []
        for j in range(len(DP)):
            temp_low = DeepPoly(expr_coef = DP[j].x_l_coef[:,i].unsqueeze(1),
                                x_l_bias = DP[j].x_l_bias[:,i].unsqueeze(1),
                                x_u_bias = DP[j].x_l_bias[:,i].unsqueeze(1))
            temp_upp = DeepPoly(expr_coef = DP[j].x_u_coef[:,i].unsqueeze(1),
                                x_l_bias = DP[j].x_u_bias[:,i].unsqueeze(1),
                                x_u_bias = DP[j].x_u_bias[:,i].unsqueeze(1))

            lb_low, ub_low = temp_low.dp_concretize(bounds = (lb_prev,ub_prev))
            lb_upp, ub_upp = temp_upp.dp_concretize(bounds = (lb_prev,ub_prev))

            constraint_low.append(temp_low)
            constraint_upp.append(temp_upp)
            concrete_bounds_low.append((lb_low, ub_low))
            concrete_bounds_upp.append((lb_upp,ub_upp))

        sort = sorted(range(len(concrete_bounds_low)), key=lambda m: concrete_bounds_low[m][0]) 
        concrete_bounds_low = [concrete_bounds_low[m] for m in sort]
        constraint_low = [constraint_low[m] for m in sort]
        sort = sorted(range(len(concrete_bounds_upp)), key=lambda m: concrete_bounds_upp[m][1],reverse = True) 
        concrete_bounds_upp = [concrete_bounds_upp[m] for m in sort]
        constraint_upp = [constraint_upp[m] for m in sort]
        while len(constraint_low)> 1:
            #since sorted accourding to increasing lowerbound, as soon as condition is true we have already minimum
            if ((concrete_bounds_low[0][1] <= concrete_bounds_low[1][0])):
                low_c_4+=1
                break
            x0, _ = constraint_low.pop(0), concrete_bounds_low.pop(0)
            x1, _ = constraint_low.pop(0), concrete_bounds_low.pop(0)
            
            lb, ub = DeepPoly(x_l_coef = x0.x_l_coef - x1.x_l_coef,
                        x_u_coef = x0.x_u_coef - x1.x_u_coef,
                        x_l_bias = x0.x_l_bias - x1.x_l_bias,
                        x_u_bias = x0.x_u_bias - x1.x_u_bias).dp_concretize((lb_prev,ub_prev))
            if ub <= 0:
                z = x0
                low_c_1 += 1
            elif lb >= 0:
                z = x1
                low_c_2 +=1
            else:
                low_c_3 += 1
                similarities.append(_sim(x0,x1,"low").item())
                z = DeepPoly(x_l_coef = (x0.x_l_coef * (-lb) + x1.x_l_coef * ub)/(ub-lb),
                            x_u_coef = (x0.x_u_coef * (-lb) + x1.x_u_coef * ub)/(ub-lb),
                            x_l_bias = (x0.x_l_bias * (-lb) + x1.x_l_bias * ub + ub * lb)/(ub-lb),
                            x_u_bias = (x0.x_u_bias * (-lb) + x1.x_u_bias * ub + ub * lb)/(ub-lb))
            constraint_low.insert(0,z)
            concrete_bounds_low.insert(0,(z.dp_concretize((lb_prev,ub_prev))))

        while len(constraint_upp)> 1:
            #since sorted accourding to decreasing upper bound, as soon as condition is true we have already maximum
            if (concrete_bounds_upp[1][1] <= concrete_bounds_upp[0][0]):
                upp_c_4 += 1
                break
            x0,_ = constraint_upp.pop(0), concrete_bounds_upp.pop(0)
            x1,_ = constraint_upp.pop(0), concrete_bounds_upp.pop(0)
            
            lb, ub = DeepPoly(x_l_coef = x0.x_l_coef - x1.x_l_coef,
                                x_u_coef = x0.x_u_coef - x1.x_u_coef,
                                x_l_bias = x0.x_l_bias - x1.x_l_bias,
                                x_u_bias = x0.x_u_bias - x1.x_u_bias).dp_concretize((lb_prev,ub_prev))
            if ub <= 0:
                z = x1
                upp_c_1 += 1
            elif lb >= 0:
                z = x0
                upp_c_2 +=1
            else:
                upp_c_3 +=1
                similarities.append(_sim(x0,x1,"upp").item())
                z = DeepPoly(x_l_coef = (x1.x_l_coef * (-lb) + x0.x_l_coef * ub)/(ub-lb),
                             x_u_coef = (x1.x_u_coef * (-lb) + x0.x_u_coef * ub)/(ub-lb),
                            x_l_bias = (x1.x_l_bias * (-lb) + x0.x_l_bias * ub - ub * lb)/(ub-lb),
                            x_u_bias = (x1.x_u_bias * (-lb) + x0.x_u_bias * ub - ub * lb)/(ub-lb))
            constraint_upp.insert(0,z)
            concrete_bounds_upp.insert(0,(z.dp_concretize((lb_prev,ub_prev))))

        DP_aggregated.x_u_coef[:,i] = constraint_upp[0].x_u_coef
        DP_aggregated.x_l_coef[:,i] = constraint_low[0].x_l_coef
        DP_aggregated.x_u_bias[:,i] = constraint_upp[0].x_u_bias
        DP_aggregated.x_l_bias[:,i] = constraint_low[0].x_l_bias



    if verbose:
        return DeepPoly(DP_aggregated.x_l_coef.view(*shp), DP_aggregated.x_u_coef.view(*shp), DP_aggregated.x_l_bias, DP_aggregated.x_u_bias), similarities

    return DeepPoly(DP_aggregated.x_l_coef.view(*shp), DP_aggregated.x_u_coef.view(*shp), DP_aggregated.x_l_bias, DP_aggregated.x_u_bias)

def _sim(x,y,mode = "low"):
    return inner_prod(x,y,mode)/(inner_prod(x,x,mode).sqrt()*inner_prod(y,y,mode).sqrt() )
def inner_prod(x,y,mode="low",lb=None,ub= None):
    if lb == None:
        if mode == "low":
            temp = (x.x_l_bias * y.x_l_bias).sum()
            temp += (x.x_l_coef * y.x_l_coef).sum()
        else:
            temp = (x.x_u_bias * y.x_u_bias).sum()
            temp += (x.x_u_coef * y.x_u_coef).sum()
        return temp
    else:
        print("not implemented yet")
        return None

   