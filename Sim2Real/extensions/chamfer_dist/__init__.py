# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import math
import torch

import chamfer


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2

class ChamferDistanceL1_PM(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, _, _, _ = ChamferFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        return torch.mean(dist1)

class DensityAwareChamferDistance(torch.nn.Module):
    f''' Density Aware Chamder Distance
    '''
    def __init__(self, alpha=1000, n_lambda=1, non_reg=False, ignore_zeros=False):
        super().__init__()
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.non_reg = non_reg
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        n_x = xyz1.size(1)
        n_gt = xyz2.size(1)


        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

        # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
        # idx1  (batch_size, n_gt): the idx of x' in [0, n_x-1]
        dist1, dist2, idx1, idx2 = ChamferFunction.apply(xyz2, xyz1)
        exp_dist1, exp_dist2 = torch.exp(-dist1 * self.alpha), torch.exp(-dist2 * self.alpha)

        count1 = torch.zeros_like(idx2)
        count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
        weight1 = count1.gather(1, idx1.long()).float().detach() ** self.n_lambda
        # weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        weight1 = torch.maximum(frac_21 / weight1 + 1e-6, torch.ones_like(weight1)) ** (-1)
        loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

        count2 = torch.zeros_like(idx1)
        count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
        weight2 = count2.gather(1, idx2.long()).float().detach() ** self.n_lambda
        # weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        weight2 = (math.ceil(frac_21) * weight2 + 1e-6) ** (-1)
        loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

        return torch.mean((loss1 + loss2) / 2)