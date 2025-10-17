import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
import time
from aim import Run, Text

def compute_mmd(x: torch.Tensor,
                y: torch.Tensor,
                kernel_mul: float = 2.0,
                kernel_num: int = 5,
                fix_sigma: float = None) -> torch.Tensor:

    """
    Gaussian-kernel MMD between two batches of features x and y.
    x, y: (batch_size, feature_dim)
    Returns a scalar MMD loss.
    """
    B = x.size(0)

    xx = x @ x.t()
    yy = y @ y.t()
    xy = x @ y.t()

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * xy

    if fix_sigma:
        sigma = fix_sigma
    else:
        sigma = torch.mean(dxx.detach())
    sigmas = [sigma * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_vals = [torch.exp(-dxx / s) + torch.exp(-dyy / s) - 2*torch.exp(-dxy / s)
               for s in sigmas]
    mmd_matrix = sum(kernel_vals)
    mmd_scalar = mmd_matrix.sum() / (kernel_num * B * B)
    return mmd_scalar

ipc_min, ipc_max = 1, 10

def get_gamma(ipc: int,
             ipc_min_val: int = ipc_min,
             ipc_max_val: int = ipc_max) -> float:
    if ipc_max_val == ipc_min_val:
        return 1.0
    γ = (ipc - ipc_min_val) / (ipc_max_val - ipc_min_val)
    return float(min(max(γ, 0.0), 1.0))

def schedule_gamma(curr_iter: int,
                   max_iter: int,
                   gamma_start: float,
                   gamma_end: float) -> float:
    if max_iter <= 0:
        return gamma_end
    frac = curr_iter / max_iter
    return float(gamma_start + frac * (gamma_end - gamma_start))

def compute_hessian(loss, model):
    """
    计算海森矩阵的近似值
    :param loss: 损失函数
    :param model: 模型
    :return: 海森矩阵（近似）
    """
    hessian = []
    params = list(model.parameters())
    for p in params:
        p_grad = torch.autograd.grad(loss, p, create_graph=True)[0]
        hessian_row = []
        for grad in p_grad.view(-1):
            grad2 = torch.autograd.grad(grad, p, create_graph=True)[0]
            hessian_row.append(grad2.view(-1))
        hessian.append(torch.cat(hessian_row))
    return torch.stack(hessian)

def compute_sample_weights_from_hessian(hessian_matrix):
    """
    从海森矩阵计算每个样本的权重
    :param hessian_matrix: 海森矩阵
    :return: 每个样本的权重
    """
    sample_weights = torch.abs(hessian_matrix).sum(dim=1)
    # 归一化权重
    sample_weights = sample_weights / sample_weights.sum()
    return sample_weights

def compute_weighted_mmd(fr_flat, fs_flat, sample_weights, kernel_mul=2.0, kernel_num=5, fix_sigma=1.0):
    """
    计算加权MMD损失
    :param fr_flat: 真实图像特征
    :param fs_flat: 合成图像特征
    :param sample_weights: 样本权重
    :return: 加权MMD损失
    """
    # 计算MMD损失
    mmd_loss = 0.0
    # 计算核函数
    for i in range(kernel_num):
        # 示例：使用高斯核进行MMD计算
        diff = fr_flat.unsqueeze(1) - fs_flat.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)
        kernel = torch.exp(-dist_sq / (2 * fix_sigma ** 2))
        mmd_loss += kernel.mean()
    
    # 加权MMD损失
    weighted_mmd_loss = (sample_weights * mmd_loss).sum()
    return weighted_mmd_loss

def tensor(l) -> torch.Tensor:
    return torch.tensor(l, requires_grad=False)

def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_

def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss

# for multiple processing
def feat_loss_for_ipc_reg(args, img_real, img_syn, model, indices, init_loop, n_iter, num_sampled) -> list:
    """
    This functions computes the feature loss of each ipcx, later used for the selection of regularized ipcx
    """

    num_samples = min(len(img_real), len(img_syn))
    img_real_gpu = img_real  
    img_syn_gpu = img_syn
    n = num_sampled 

    if n > num_samples:
        real_indices = torch.randperm(img_real_gpu.size(0), device=img_real_gpu.device)[:num_samples]  
        sampled_real_imgs = img_real_gpu[real_indices]  
        sampled_syn_imgs = img_syn_gpu[:num_samples] 
    else:
        sampled_real_imgs = img_real_gpu  
        syn_indices = torch.randperm(img_syn_gpu.size(0), device=img_syn_gpu.device)[:num_samples] 
        sampled_syn_imgs = img_syn_gpu[syn_indices]
    # with torch.no_grad():
    #     feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
    # feat = model.get_feature(img_syn, args.idx_from, args.idx_to)
    with torch.no_grad():
        feats_real = model.get_feature(sampled_real_imgs, args.idx_from, args.idx_to)
        feats_holistic = model.get_feature(img_real_gpu, args.idx_from, args.idx_to)
    feats_syn = model.get_feature(sampled_syn_imgs, args.idx_from, args.idx_to)    

    loss_list = []

    for ipcx in args.adaptive_reg_list:
        # loss = None
        loss_mse = None
        loss_mmd = 0.0
        alpha = 0

        for i in range(len(feats_syn)):
            feat_ipcx = feats_syn[i][indices[ipcx]]
            loss_mse = add_loss(loss_mse, dist(alpha * feats_real[i].mean(0) + (1 - alpha) * feats_holistic[i].mean(0), feat_ipcx.mean(0), method=args.metric))
 
        for fr, fs in zip(feats_real, feats_syn):
            fr_flat = fr.reshape(fr.size(0), -1)
            fs_flat = fs.reshape(fs.size(0), -1)
            loss_mmd += compute_mmd(
                fs_flat,
                fr_flat,
                kernel_mul=args.mmd_kernel_mul,
                kernel_num=args.mmd_kernel_num,
                fix_sigma=args.mmd_fix_sigma
            )

        if args.gamma_schedule:
            γ = schedule_gamma(curr_iter=init_loop, max_iter=n_iter - 1, gamma_start=args.gamma_start, gamma_end=args.gamma_end)
        elif args.gamma is not None:
            γ = args.gamma
        else:
            γ = get_gamma(args.ipc)

        γ = max(γ, args.min_mse)
        γ = min(γ, 1.0 - args.min_mmd)
        loss = γ * loss_mse + (1.0 - γ) * loss_mmd
        
        loss_list.append(loss.item())
        
    
    assert len(loss_list) == len(args.adaptive_reg_list)
    return loss_list

# for multiple processing
def grad_loss_for_img_update(args, img_real, img_syn, lab_real, lab_syn, model, ipcx_list, indices, num_sampled, init_loop, n_iter):
    """
    This functions computes the gradient loss for the image update od each ipcx
    """
    if args.match != 'grad':
        return NotImplementedError
    num_samples = min(len(img_real), len(img_syn))
    img_real_gpu = img_real  
    img_syn_gpu = img_syn
    n = num_sampled 

    if n > num_samples:
        real_indices = torch.randperm(img_real_gpu.size(0), device=img_real_gpu.device)[:num_samples]  
        sampled_real_imgs = img_real_gpu[real_indices]  
        sampled_syn_imgs = img_syn_gpu[:num_samples] 
    else:
        sampled_real_imgs = img_real_gpu  
        syn_indices = torch.randperm(img_syn_gpu.size(0), device=img_syn_gpu.device)[:num_samples] 
        sampled_syn_imgs = img_syn_gpu[syn_indices]
    with torch.no_grad():
        feats_real = model.get_feature(sampled_real_imgs, args.idx_from, args.idx_to)
    feats_syn = model.get_feature(sampled_syn_imgs, args.idx_from, args.idx_to)   

    loss_mse = None
    loss_mmd = 0

    criterion = nn.CrossEntropyLoss()

    output_real = model(img_real)
    loss_real = criterion(output_real, lab_real)
    g_real = torch.autograd.grad(loss_real, model.parameters())

    # import pdb; pdb.set_trace()
    g_real = list((g.detach() for g in g_real))

    # change all functions to time_func
    output_syn = model(img_syn)
    loss_syn = criterion(output_syn, lab_syn)
    g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

    ipcx_dict = {}
    for ipcx in ipcx_list:
        ipcx_indices = tensor(indices[ipcx])
        # compute the gradient of the first ipc1 images
        syn_ipcx = img_syn[ipcx_indices]
        lab_ipcx = lab_syn[ipcx_indices]
        ipcx_dict[ipcx] = {'syn_ipcx': syn_ipcx, 'lab_ipcx': lab_ipcx}

        output_syn_ipcx = model(syn_ipcx)
        loss_syn_ipcx = criterion(output_syn_ipcx, lab_ipcx)
        g_syn_ipcx = torch.autograd.grad(loss_syn_ipcx, model.parameters(), create_graph=True)
        ipcx_dict[ipcx].update({'output_syn_ipcx': output_syn_ipcx, 'g_syn_ipcx': g_syn_ipcx, 'loss_syn_ipcx': loss_syn_ipcx.item()})

    for i in range(len(g_real)):
        if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
            continue
        if (len(g_real[i].shape) == 2) and not args.fc:
            continue

        loss_mse = add_loss(loss_mse, dist(g_real[i], g_syn[i], method=args.metric))

        if ipcx_list is not None:
            # compute the regularization term
            match_term = g_real[i]

            # add multiple ipcx loss
            for ipcx in ipcx_list:
                loss_mse = add_loss(loss_mse, dist(match_term, ipcx_dict[ipcx]['g_syn_ipcx'][i], method=args.metric))
 
        for fr, fs in zip(feats_real, feats_syn):
            fr_flat = fr.reshape(fr.size(0), -1)
            fs_flat = fs.reshape(fs.size(0), -1)
            loss_mmd += compute_mmd(
                fs_flat,
                fr_flat,
                kernel_mul=args.mmd_kernel_mul,
                kernel_num=args.mmd_kernel_num,
                fix_sigma=args.mmd_fix_sigma
            )

        if args.gamma_schedule:
            γ = schedule_gamma(curr_iter=init_loop, max_iter=n_iter - 1, gamma_start=args.gamma_start, gamma_end=args.gamma_end)
        elif args.gamma is not None:
            γ = args.gamma
        else:
            γ = get_gamma(args.ipc)

        γ = max(γ, args.min_mse)
        γ = min(γ, 1.0 - args.min_mmd)
        loss = γ * loss_mse + (1.0 - γ) * loss_mmd

    return loss

def select_reg_ipc(args, regularizer, it, logger=None, aim_run=None) -> list:
    """
    Select current ipcx based on the feature loss
    """
    if (it == 1) or (len(regularizer.stats["prev_loss"]) == 0):
        assert len(args.adaptive_reg_list) > 0, "adaptive_reg_list should not be empty"
        selected_ipcx = args.adaptive_reg_list[0]
        return [selected_ipcx]
    else:
        assert len(regularizer.stats["prev_loss"]) == len(args.adaptive_reg_list)   # they should have same length

        selected_ipcx_list = []

        # find the largest absolute difference between the current loss and the previous loss
        diff = torch.abs(tensor(regularizer.stats["prev_loss"]) - tensor(regularizer.stats["cur_loss"]))
        # find normalized difference
        diff_idx_list = torch.argsort(diff, descending=True)  # find the one with largest difference

        # # use concatentation to avoid list of list
        # selected_ipcx = tensor(args.adaptive_reg_list)[diff_idx_list[0]].tolist()
        # if type(selected_ipcx) == int:
        #     selected_ipcx_list += [selected_ipcx]
        # else:
        #     selected_ipcx_list += selected_ipcx
        k = min(args.adaptive_k, len(diff_idx_list))
        topk_ipcs = tensor(args.adaptive_reg_list)[diff_idx_list[:k]].tolist()

        if isinstance(topk_ipcs, int):
            selected_ipcx_list += [topk_ipcs]
        else:
            selected_ipcx_list += topk_ipcs

        all_ipcs = args.adaptive_reg_list
        rr = regularizer.rr_pointer % len(all_ipcs)
        candidate = all_ipcs[rr]
        if candidate not in selected_ipcx_list:
            selected_ipcx_list.append(candidate)
        regularizer.rr_pointer += 1        

        if aim_run:
            aim_run.track(Text(f"{torch.round(diff, decimals=2).tolist()}"), name="loss_list", step=it, context={"subset": "diff"})
        if logger:
            logger(f"it: {it}, diff: {torch.round(diff, decimals=2).tolist()}")

    assert len(selected_ipcx_list) > 0
    return selected_ipcx_list

class RegularizedIPC():
    """
    Basic class for regularized ipcx
    """
    def __init__(self, ipcx=-1, iteration=-1, regularize=False, keep_freeze=False):
        self.ipcx = ipcx
        self.iteration = iteration
        self.regularize = regularize
        self.keep_freeze = keep_freeze

class Regularizer():
    """
    The class that handles the regularization of the ipcx
    """
    def __init__(self, args):
        self.args = args
        self.reg_list = [RegularizedIPC(i) for i in range(0, args.ipc + 1)] # zero is padded to align ipcx with index
        self.prev_ipcx_list = []
        self.history = []   # store the history of regularized ipcx

        # stats for determining regularized ipcx
        # prev_loss stores the loss of the previous feature loss evaluation
        # cur_loss stores the loss of the current feature loss evaluation
        self.stats = {"prev_loss": [], "total_loss": []}
        self.rr_pointer = 0

    def __call__(self):
        """
        Default call function: return True if there is any ipcx that needs to be regularized
        """
        return len(self.get_regularized_ipc()) > 0

    def get_regularized_ipc(self)->list:
        return [item.ipcx for item in self.reg_list if item.regularize]
    
    def regularize_ipcx(self, ipcx, remove=False, prev=False):
        """
        remove: if True, remove ipcx from regularize list
        prev: if True, add ipcx to prev_ipcx_list
        """
        # set regularize to False for ipcx if not remove
        self.reg_list[ipcx].regularize = not remove
        if prev:
            self.prev_ipcx_list.append(ipcx)
    
    def update_ipc_prev_list(self):
        for ipcx_prev in self.prev_ipcx_list:    # remove previous regularized ipcx
            self.regularize_ipcx(ipcx_prev, remove=True)    # remove regularize mark
        self.prev_ipcx_list = []    # clear prev_ipcx_list
    
    def update_status(self, iteration):
        # if iteration is in stop iteration
        # set keep_freeze to True
        # set regularize to False

        update_flag = False
        for item in self.reg_list:
            if iteration == item.iteration:
                item.regularize = False
                item.keep_freeze = True
                update_flag = True
        
        return update_flag
    
    def set_quit_iteration(self, ipcx, iteration):
        self.reg_list[ipcx].iteration = iteration

    def freeze_ipcx(self, ipcx, unfreeze=False):
        if unfreeze:    # if unfreeze, unfreeze all ipcx after ipcx
            for ipcx_unfreeze in range(ipcx, self.args.ipc):
                self.reg_list[ipcx_unfreeze].keep_freeze = False
        
        if not unfreeze:
            self.reg_list[ipcx].keep_freeze = True
    
    def get_freeze_ipc(self):
        """
        Find max ipcx to freeze which is smaller than min(current ipcx)
        """
        # find all ipcx that has keep_freeze set to True
        current_ipcx = self.get_regularized_ipc()
        if len(current_ipcx) == 0:
            return -1

        ipcx_keep_freeze = []
        for item in self.reg_list:
            if item.keep_freeze and (item.ipcx < min(current_ipcx)):
                ipcx_keep_freeze.append(item.ipcx)
        
        return max(ipcx_keep_freeze) if len(ipcx_keep_freeze) > 0 else -1
    
    def print_status(self, view_all=False):
        print("Regularizer status:")
        for item in self.reg_list:
            # print with paddings for better alignment
            if view_all or item.regularize:
                print(f"ipcx: {item.ipcx:2d}, regularize: {str(item.regularize):5s}, keep_freeze: {str(item.keep_freeze):5s}, stop_iteration: {item.iteration:4d}")