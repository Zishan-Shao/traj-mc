import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
index = 0
@torch.no_grad()
def part_mean(tensor, op='-'):
    non_zero = tensor*(tensor!=0)

    mean_val = non_zero.mean(-1).view(-1, 1)

    return mean_val

@torch.no_grad()
def high_order_residual(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    return sum_order

@torch.no_grad()
def high_order_residual_rc(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # mean row
        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        masked_x_tensor -= mean_tensor_all_r[:, None]
        # mean column
        mean_tensor_all_c = torch.nanmean(masked_x_tensor, dim=0)
        mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        masked_x_tensor -= mean_tensor_all_c[None, :]

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        binary += mean_tensor_all_r[:, None] + mean_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    return sum_order

@torch.no_grad()
def high_order_residual_alternating_order1(x, mask, order=2, iter=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()

    for k in range(iter):
        # 1. Fix alpha and B, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()

        # 2. Fix mean and B, update alpha
        new_alpha = 1. / (torch.sum(new_binary * mask * new_binary * mask, dim=1) + 1e-8) * torch.sum(new_binary * mask * (new_matrix - refine_mean[:, None] * mask), dim=1)

        # 3. Fix mean and alpha, update B
        new_binary = torch.sign(new_matrix - refine_mean[:, None] * mask)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask


    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order1_x(x, mask, order=2, S=None, iter=15, iter2=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()
    new_alpha = scale_tensor_all.clone()

    for k in range(iter):
        # 1. Fix alpha and B, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()

        # 2. Fix mean and B, update alpha
        new_alpha = 1. / (torch.sum(new_binary * mask * new_binary * mask, dim=1) + 1e-8) * torch.sum(new_binary * mask * (new_matrix - refine_mean[:, None] * mask), dim=1)

        # 3. Fix mean and alpha, update B
        new_binary = torch.sign(new_matrix - refine_mean[:, None] * mask)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = torch.sum(S * MM, dim=(1,2), dtype=torch.float32) + 1e-10
    masked_B = new_binary * mask
    new_alpha_den = torch.sum(S * masked_B[:, :, None] * masked_B[:, None, :], dim=(1,2)) + 1e-10
    # diag_S = torch.diag(S)
    for kk in range(iter2):
        # X error update mean
        refine_mean = torch.sum(S * (new_matrix - new_alpha[:, None] * new_binary * mask)[:, :, None] * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha
        new_alpha = torch.sum(S * masked_B[:, :, None] * (new_matrix - refine_mean[:, None] * mask)[:, None, :], dim=(1,2)) / new_alpha_den

    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order3_rc_nomean(x, mask, order=3, iter=15):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list_r = []
    alpha_list_c = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        alpha_list_r.append(scale_tensor_all_r.clone())
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        alpha_list_c.append(scale_tensor_all_c.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()

    for k in range(iter):
        # 2-1. Fix mean, alpha column, and B, update alpha row 0
        W_tilde = new_matrix - (alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1] +
                                alpha_list_c[2][None, :] * alpha_list_r[2][:, None] * binary_list[2]) * mask
        alpha_c_B = alpha_list_c[0][None, :] * binary_list[0] * mask
        alpha_list_r[0] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)

        # 2-2. Fix mean, alpha row, and B, update alpha column 0
        alpha_r_B =  alpha_list_r[0][:, None] * binary_list[0] * mask
        alpha_list_c[0] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 2-3. Fix mean, alpha column, and B, update alpha row 1
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] +
                                alpha_list_c[2][None, :] * alpha_list_r[2][:, None] * binary_list[2]) * mask
        alpha_c_B = alpha_list_c[1][None, :] * binary_list[1] * mask
        alpha_list_r[1] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)

        # 2-4. Fix mean, alpha row, and B, update alpha column 1
        alpha_r_B =  alpha_list_r[1][:, None] * binary_list[1] * mask
        alpha_list_c[1] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 2-5. Fix mean, alpha column, and B, update alpha row 2
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] +
                                alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
        alpha_c_B = alpha_list_c[2][None, :] * binary_list[2] * mask
        alpha_list_r[2] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)

        # 2-6. Fix mean, alpha row, and B, update alpha column 2
        alpha_r_B =  alpha_list_r[2][:, None] * binary_list[2] * mask
        alpha_list_c[2] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 3. Fix mean and alpha, update B
        new_matrix_expanded = new_matrix.unsqueeze(-1)
        comb0 = alpha_list_r[0].reshape(-1, 1) @ alpha_list_c[0].reshape(1, -1)
        comb1 = alpha_list_r[1].reshape(-1, 1) @ alpha_list_c[1].reshape(1, -1)
        comb2 = alpha_list_r[2].reshape(-1, 1) @ alpha_list_c[2].reshape(1, -1)
        v = torch.stack([-comb0 - comb1 - comb2, -comb0 - comb1 + comb2,
                         -comb0 + comb1 - comb2, -comb0 + comb1 + comb2,
                         +comb0 - comb1 - comb2, +comb0 - comb1 + comb2,
                         +comb0 + comb1 - comb2, +comb0 + comb1 + comb2], dim=2)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1) | (min_indices == 2) | (min_indices == 3)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 1) | (min_indices == 4) | (min_indices == 5)] = -1
        binary_list[2] = torch.ones_like(min_indices)
        binary_list[2][(min_indices == 0) | (min_indices == 2) | (min_indices == 4) | (min_indices == 6)] = -1

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] +
                                                      alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1] +
                                                      alpha_list_c[2][None, :] * alpha_list_r[2][:, None] * binary_list[2]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order2_rc_nomean(x, mask, order=2, iter=15):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list_r = []
    alpha_list_c = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        alpha_list_r.append(scale_tensor_all_r.clone())
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        alpha_list_c.append(scale_tensor_all_c.clone())

        binary= torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()

    for k in range(iter):
        # 2-1. Fix mean, alpha column, and B, update alpha row 0
        W_tilde = new_matrix - (alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
        alpha_c_B = alpha_list_c[0][None, :] * binary_list[0] * mask
        alpha_list_r[0] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)

        # 2-2. Fix mean, alpha row, and B, update alpha column 0
        alpha_r_B =  alpha_list_r[0][:, None] * binary_list[0] * mask
        alpha_list_c[0] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 2-3. Fix mean, alpha column, and B, update alpha row 1
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0]) * mask
        alpha_c_B = alpha_list_c[1][None, :] * binary_list[1] * mask
        alpha_list_r[1] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)

        # 2-4. Fix mean, alpha row, and B, update alpha column 1
        alpha_r_B =  alpha_list_r[1][:, None] * binary_list[1] * mask
        alpha_list_c[1] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 3. Fix mean and alpha, update B
        new_matrix_expanded = new_matrix.unsqueeze(-1)
        comb0 = alpha_list_r[0].reshape(-1, 1) @ alpha_list_c[0].reshape(1, -1)
        comb1 = alpha_list_r[1].reshape(-1, 1) @ alpha_list_c[1].reshape(1, -1)
        v = torch.stack([-comb0 - comb1, -comb0 + comb1,
                    comb0 - comb1, comb0 + comb1], dim=2)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] + alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order1_rc_nomean(x, mask, order=1, iter=15):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()
    for k in range(iter):
        # 1-1. Fix mean, alpha column, and B, update alpha row
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * new_matrix, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)

        # 1-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * new_matrix, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_mean(x, mask, order=2, num_iters=15):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0], device=x.device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    new_matrix = x.clone() * mask
    sum_order_alternating = sum_order.clone()

    for k in range(num_iters):
        # 1. Fix alpha1, alpha2, B1, and B2, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()

        # 2. Fix mean, B1, and B2, update alpha1 and alpha2
        alpha_list[0] = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
        alpha_list[1] = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)

        # 3. Fix mean, alpha1, and alpha2, update B1 and B2
        new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        v = torch.stack([-alpha_list[0] - alpha_list[1], -alpha_list[0] + alpha_list[1],
                    alpha_list[0] - alpha_list[1], alpha_list[0] + alpha_list[1]], dim=1).unsqueeze(1)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1

        sum_order_alternating = torch.zeros_like(x) + (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1] + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_mean_x(x, mask, order=2, S=None, num_iters=15, iter2=15):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0], device=x.device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    new_matrix = x.clone() * mask
    sum_order_alternating = sum_order.clone()

    for k in range(num_iters):
        # 1. Fix alpha1, alpha2, B1, and B2, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()

        # 2. Fix mean, B1, and B2, update alpha1 and alpha2
        alpha_list[0] = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
        alpha_list[1] = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)

        # 3. Fix mean, alpha1, and alpha2, update B1 and B2
        new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        v = torch.stack([-alpha_list[0] - alpha_list[1], -alpha_list[0] + alpha_list[1],
                    alpha_list[0] - alpha_list[1], alpha_list[0] + alpha_list[1]], dim=1).unsqueeze(1)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1

        sum_order_alternating = torch.zeros_like(x) + (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1] + refine_mean[:, None]) * mask

    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = torch.sum(S * MM, dim=(1,2)) + 1e-10
    masked_B0 = binary_list[0] * mask
    new_alpha0_den = torch.sum(S * masked_B0[:, :, None] * masked_B0[:, None, :], dim=(1,2)) + 1e-10
    masked_B1 = binary_list[1] * mask
    new_alpha1_den = torch.sum(S * masked_B1[:, :, None] * masked_B1[:, None, :], dim=(1,2)) + 1e-10
    for kk in range(iter2):
        # X error update mean
        refine_mean = torch.sum(S * (new_matrix - (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1]) * mask)[:, :, None] * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha
        masked_W_mu = new_matrix - refine_mean[:, None] * mask
        alpha_list[0] = torch.sum(S * masked_B0[:, :, None] * (masked_W_mu[:, None, :] - (alpha_list[1][:, None] * masked_B1)[:, None, :]), dim=(1,2)) / new_alpha0_den
        alpha_list[1] = torch.sum(S * masked_B1[:, :, None] * (masked_W_mu[:, None, :] - (alpha_list[0][:, None] * masked_B0)[:, None, :]), dim=(1,2)) / new_alpha1_den

    sum_order_alternating = torch.zeros_like(x) + (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1] + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_mean_x_new(x, mask, order=2, S=None):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    device = x.device
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    global index
    index += 1

    # 初始化
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0], device=device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=device))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())
        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary = binary * scale_tensor_all[:, None] + mean_tensor_all[:, None]
        sum_order = sum_order + binary * mask

    # 闭式解部分
    # 只支持 order=2
    assert order == 2, "闭式解目前只支持order=2"
    B0 = binary_list[0]
    B1 = binary_list[1]
    M = mask
    T = x.shape[0]
    N = x.shape[1]
    if S is None:
        S = torch.eye(N, device=device)

    # 构造闭式解所需的矩阵
    one = torch.ones((S.shape[0], 1), device=device)
    # 计算各项
    a = torch.sum((B0 @ S) * B0, dim=1)
    b = torch.sum((B0 @ S) * B1, dim=1)
    c = torch.sum((B1 @ S) * B1, dim=1)
    d = (one.T @ S @ one).squeeze()
    e = torch.sum((B0 @ S) * one.T, dim=1)
    f = torch.sum((B1 @ S) * one.T, dim=1)
    y1 = torch.sum((x @ S) * B0, dim=1)
    y2 = torch.sum((x @ S) * B1, dim=1)
    y3 = torch.sum((x @ S) * one.T, dim=1)

    # 组装线性方程组
    # [a  b  e] [alpha0]   [y1]
    # [b  c  f] [alpha1] = [y2]
    # [e  f  d] [ mu   ]   [y3]
    # 求解每一行的参数
    A = torch.stack([
        torch.stack([a, b, e], dim=1),
        torch.stack([b, c, f], dim=1),
        torch.stack([e, f, d.expand_as(e)], dim=1)
    ], dim=1)  # [T, 3, 3]
    Y = torch.stack([y1, y2, y3], dim=1).unsqueeze(-1)  # [T, 3, 1]
    # 闭式解
    params = torch.linalg.solve(A, Y).squeeze(-1)  # [T, 3]
    alpha0 = params[:, 0]
    alpha1 = params[:, 1]
    mu = params[:, 2]

    # 组装最终结果
    sum_order_alternating = (alpha0[:, None] * B0 + alpha1[:, None] * B1 + mu[:, None]) * mask
    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_mean_x_new_revised(x, mask, Hinv_diag, order=2, S=None, num_iters=15, lambda_salience=2.0):
    """
    结合迭代优化和闭式解：
    1. 初始化 B0, B1, alpha, mean。
    2. 进行 num_iters 次迭代，交替优化 B0, B1, alpha, mean，以获得更好的基矩阵。
    3. 使用迭代优化后的 B0, B1，通过一次性闭式解精确计算最终的 alpha0, alpha1, mu。
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    device = x.device
    # 1. 生成显著性掩码和加权矩阵
    mask_non_salient, mask_salient = saliency_mask_from_hessian(x, Hinv_diag, threshold=2.0)
    #_, mask_salient2 = saliency_mask_from_hessian(x, Hinv_diag, threshold=3.0)
    # 创建一个静态的误差加权矩阵，显著位置的权重更高，在计算最小二乘解时，误差项会被平方，所以这里直接用lambda
    error_weights = torch.ones_like(x)
    error_weights[mask_salient] = 3
    #error_weights[mask_salient2] = 3

    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    global index
    index += 1

    # --- 1. 初始化 ---
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0], device=device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=device))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())
        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary = binary * scale_tensor_all[:, None] + mean_tensor_all[:, None]
        sum_order = sum_order + binary * mask

    # --- 2. 迭代优化 ---
    sum_order_alternating = sum_order.clone()
    # for k in range(num_iters):
    #     # 2.1. 更新均值 (mean)
    #     residual = new_matrix - sum_order_alternating
    #     masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=device))
    #     mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
    #     mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
    #     refine_mean += mean_tensor_all.clone()

    #     # 2.2. 更新 alpha
    #     alpha_list[0] = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
    #     alpha_list[1] = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)

    #     # 2.3. 更新二值基 B
    #     new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
    #     v = torch.stack([-alpha_list[0] - alpha_list[1], -alpha_list[0] + alpha_list[1],
    #                 alpha_list[0] - alpha_list[1], alpha_list[0] + alpha_list[1]], dim=1).unsqueeze(1)
    #     min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)
    #     binary_list[0] = torch.ones_like(min_indices)
    #     binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
    #     binary_list[1] = torch.ones_like(min_indices)
    #     binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1

    #     sum_order_alternating = (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1] + refine_mean[:, None]) * mask
    for k in range(num_iters):
        # 1. 更新 mean
        residual = new_matrix - sum_order_alternating
        mean_tensor_all = torch.sum(residual * error_weights, dim=1) / (torch.sum(error_weights, dim=1) + 1e-8)  # [MOD]
        refine_mean += mean_tensor_all.clone()

        # 2. 更新 alpha1 和 alpha2 (加权最小二乘)
        target0 = new_matrix - refine_mean[:, None] * mask - alpha_list[1][:, None] * binary_list[1] * mask
        num0 = torch.sum(binary_list[0] * target0 * error_weights, dim=1)               # [MOD]
        den0 = torch.sum((binary_list[0]**2) * error_weights, dim=1) + 1e-8              # [MOD]
        alpha_list[0] = num0 / den0                                                     # [MOD]

        target1 = new_matrix - refine_mean[:, None] * mask - alpha_list[0][:, None] * binary_list[0] * mask
        num1 = torch.sum(binary_list[1] * target1 * error_weights, dim=1)               # [MOD]
        den1 = torch.sum((binary_list[1]**2) * error_weights, dim=1) + 1e-8              # [MOD]
        alpha_list[1] = num1 / den1                                                     # [MOD]

        # 3. 更新 B（比较时乘 error_weights）
        new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        v = torch.stack([
            -alpha_list[0] - alpha_list[1],
            -alpha_list[0] + alpha_list[1],
             alpha_list[0] - alpha_list[1],
             alpha_list[0] + alpha_list[1]
        ], dim=1).unsqueeze(1)

        weighted_diff = torch.abs(new_matrix_expanded - v) * error_weights.unsqueeze(-1)  # [MOD]
        min_indices = torch.argmin(weighted_diff, dim=-1)                                 # [MOD]

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1

        sum_order_alternating = (alpha_list[0][:, None] * binary_list[0] +
                                 alpha_list[1][:, None] * binary_list[1] +
                                 refine_mean[:, None]) * mask

    # --- 3. 闭式解 (替换 iter2) ---
    assert order == 2, "闭式解目前只支持order=2"
    B0 = binary_list[0].to(x.dtype)
    B1 = binary_list[1].to(x.dtype)
    N = x.shape[1]
    if S is None:
        S = torch.eye(N, device=device)

    one = torch.ones((S.shape[0], 1), device=device)
    a = torch.sum((B0 @ S) * B0, dim=1)
    b = torch.sum((B0 @ S) * B1, dim=1)
    c = torch.sum((B1 @ S) * B1, dim=1)
    d = (one.T @ S @ one).squeeze()
    e = torch.sum((B0 @ S) * one.T, dim=1)
    f = torch.sum((B1 @ S) * one.T, dim=1)
    y1 = torch.sum((x @ S) * B0, dim=1)
    y2 = torch.sum((x @ S) * B1, dim=1)
    y3 = torch.sum((x @ S) * one.T, dim=1)

    A = torch.stack([
        torch.stack([a, b, e], dim=1),
        torch.stack([b, c, f], dim=1),
        torch.stack([e, f, d.expand_as(e)], dim=1)
    ], dim=1)
    Y = torch.stack([y1, y2, y3], dim=1).unsqueeze(-1)

    # 为防止 A 不可逆，添加一个小的扰动项
    diagonals = torch.diagonal(A, dim1=-2, dim2=-1)
    damp = 0.01 * torch.mean(diagonals)
    A.diagonal(dim1=-2, dim2=-1).add_(damp)
    params = torch.linalg.solve(A, Y).squeeze(-1)

    alpha0 = params[:, 0]
    alpha1 = params[:, 1]
    mu = params[:, 2]

    result = (alpha0[:, None] * B0 + alpha1[:, None] * B1 + mu[:, None]) * mask
    return result

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

@torch.no_grad()
def saliency_mask_from_hessian(x, Hinv_diag, threshold=2.0):
    """
    完全按照 SliM-LLM 的方法生成显著性掩码。
    不使用百分比阈值，而是使用统计异常值检测。

    Args:
        x (torch.Tensor): 模型的权重张量。
        Hinv_diag (torch.Tensor): Hessian逆矩阵的对角线元素，应与x具有相同的元素数量。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - mask_non_salient (torch.Tensor): 非显著元素的布尔掩码。
            - mask_salient (torch.Tensor): 显著元素的布尔掩码。
    """

    # 1. 计算逐元素的敏感度 (基于Optimal Brain Surgeon/Damage方法)
    # 参考SliM-LLM的计算方式：weight^2 / diag(H^{-1})^2
    sensitivity = x.pow(2) / (Hinv_diag.pow(2) + 1e-12)

    # 2. 完全按照 SliM-LLM 的统计方法确定阈值
    #threshold = 2.0  # SliM-LLM 中使用的固定阈值

    mean_sensitivity = torch.mean(sensitivity)
    std_sensitivity = torch.std(sensitivity)
    z_scores = (sensitivity - mean_sensitivity) / (std_sensitivity + 1e-12)

    # 3. 找出统计异常值
    outliers = sensitivity[torch.abs(z_scores) > threshold]

    if outliers.numel() > 0:
        # 使用异常值的最小值作为阈值
        outlier_min_value = torch.min(outliers)
        mask_salient = sensitivity >= outlier_min_value
    else:
        # 如果没有统计异常值，则没有显著权重
        mask_salient = torch.zeros_like(sensitivity, dtype=torch.bool)

    mask_non_salient = ~mask_salient

    return mask_non_salient, mask_salient

@torch.no_grad()
def saliency_weights_from_hessian_sigmoid(x, Hinv_diag):
    """
    使用 sigmoid 函数根据 Hessian 信息生成连续的显著性权重。
    权重值是连续的，而不是离散的。

    Args:
        x (torch.Tensor): 模型的权重张量。
        Hinv_diag (torch.Tensor): Hessian逆矩阵的对角线元素。
        k (float): Sigmoid 函数的缩放/陡峭度参数。
        base_weight (float): 非显著权重的基准权重。
        peak_weight (float): 最显著权重的峰值权重。

    Returns:
        torch.Tensor: 误差加权矩阵 (error_weights)。
    """
    # 1. 计算逐元素的敏感度
    sensitivity = x.pow(2) / (Hinv_diag.pow(2) + 1e-12)

    # 2. 应用 sigmoid 函数
    sigmoid_values = torch.sigmoid(sensitivity)

    # 3. 将 sigmoid 输出 [0, 1] 映射到 [base_weight, peak_weight]
    error_weights = 1 + sigmoid_values

    return error_weights

@torch.no_grad()
def high_order_residual_alternating_order3_rc_nomean_hessian(x, mask, Hinv_diag, order=3, iter=15, lambda_salience=2.0):
    """
    Hessian-aware version of order-3 row-column alternating optimization.
    Prioritizes salient weights during optimization.
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)

    # 1. 生成显著性掩码和加权矩阵
    mask_non_salient, mask_salient = saliency_mask_from_hessian(x, Hinv_diag)
    error_weights = torch.ones_like(x)
    error_weights[mask_salient] = lambda_salience

    # 2. 初始化参数 (与原函数相同)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list_r = []
    alpha_list_c = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        alpha_list_r.append(scale_tensor_all_r.clone())
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / (scale_tensor_all_r[:, None] + 1e-12)), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        alpha_list_c.append(scale_tensor_all_c.clone())
        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # 3. 交替优化 (引入加权)
    for k in range(iter):
        # --- 更新 alpha for order 0 ---
        W_tilde = new_matrix - (alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1] +
                                alpha_list_c[2][None, :] * alpha_list_r[2][:, None] * binary_list[2]) * mask
        alpha_c_B = alpha_list_c[0][None, :] * binary_list[0] * mask
        alpha_list_r[0] = torch.sum(alpha_c_B * W_tilde * error_weights, dim=1) / (torch.sum(alpha_c_B.pow(2) * error_weights, dim=1) + 1e-8)
        alpha_r_B =  alpha_list_r[0][:, None] * binary_list[0] * mask
        alpha_list_c[0] = torch.sum(alpha_r_B * W_tilde * error_weights, dim=0) / (torch.sum(alpha_r_B.pow(2) * error_weights, dim=0) + 1e-8)

        # --- 更新 alpha for order 1 ---
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] +
                                alpha_list_c[2][None, :] * alpha_list_r[2][:, None] * binary_list[2]) * mask
        alpha_c_B = alpha_list_c[1][None, :] * binary_list[1] * mask
        alpha_list_r[1] = torch.sum(alpha_c_B * W_tilde * error_weights, dim=1) / (torch.sum(alpha_c_B.pow(2) * error_weights, dim=1) + 1e-8)
        alpha_r_B =  alpha_list_r[1][:, None] * binary_list[1] * mask
        alpha_list_c[1] = torch.sum(alpha_r_B * W_tilde * error_weights, dim=0) / (torch.sum(alpha_r_B.pow(2) * error_weights, dim=0) + 1e-8)

        # --- 更新 alpha for order 2 ---
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] +
                                alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
        alpha_c_B = alpha_list_c[2][None, :] * binary_list[2] * mask
        alpha_list_r[2] = torch.sum(alpha_c_B * W_tilde * error_weights, dim=1) / (torch.sum(alpha_c_B.pow(2) * error_weights, dim=1) + 1e-8)
        alpha_r_B =  alpha_list_r[2][:, None] * binary_list[2] * mask
        alpha_list_c[2] = torch.sum(alpha_r_B * W_tilde * error_weights, dim=0) / (torch.sum(alpha_r_B.pow(2) * error_weights, dim=0) + 1e-8)

        # --- 更新 B (二值矩阵) ---
        # [MODIFIED] 引入加权误差来寻找最近的二值组合
        new_matrix_expanded = new_matrix.unsqueeze(-1)
        comb0 = alpha_list_r[0].reshape(-1, 1) @ alpha_list_c[0].reshape(1, -1)
        comb1 = alpha_list_r[1].reshape(-1, 1) @ alpha_list_c[1].reshape(1, -1)
        comb2 = alpha_list_r[2].reshape(-1, 1) @ alpha_list_c[2].reshape(1, -1)
        v = torch.stack([-comb0 - comb1 - comb2, -comb0 - comb1 + comb2,
                         -comb0 + comb1 - comb2, -comb0 + comb1 + comb2,
                         +comb0 - comb1 - comb2, +comb0 - comb1 + comb2,
                         +comb0 + comb1 - comb2, +comb0 + comb1 + comb2], dim=2)

        # 误差项乘以权重，使得显著位置的误差在比较中更重要
        weighted_error = torch.abs(new_matrix_expanded - v) * error_weights.unsqueeze(-1)
        min_indices = torch.argmin(weighted_error, dim=-1)

        # 从 min_indices 重构二值矩阵的逻辑保持不变
        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1) | (min_indices == 2) | (min_indices == 3)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 1) | (min_indices == 4) | (min_indices == 5)] = -1
        binary_list[2] = torch.ones_like(min_indices)
        binary_list[2][(min_indices == 0) | (min_indices == 2) | (min_indices == 4) | (min_indices == 6)] = -1

    # 4. Final refine results (与原函数相同)
    sum_order_alternating = (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] +
                             alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1] +
                             alpha_list_c[2][None, :] * alpha_list_r[2][:, None] * binary_list[2]) * mask
    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order2_rc_nomean_hessian(x, mask, Hinv_diag, order=2, iter=15, lambda_salience=2.0):
    """
    修改后的版本：
    - 保持二阶和行列scale参数量不变。
    - 依据Hessian信息指导迭代公式，优先优化显著权重。
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)

    # 1. 生成显著性掩码和加权矩阵
    mask_non_salient, mask_salient = saliency_mask_from_hessian(x, Hinv_diag)

    # 创建一个静态的误差加权矩阵，显著位置的权重更高
    # 在计算最小二乘解时，误差项会被平方，所以这里直接用lambda
    error_weights = torch.ones_like(x)
    error_weights[mask_salient] = lambda_salience

    # error_weights = saliency_weights_from_hessian_sigmoid(x, Hinv_diag)

    # 2. 初始化参数 (与原函数相同)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list_r = []
    alpha_list_c = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        alpha_list_r.append(scale_tensor_all_r.clone())
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / (scale_tensor_all_r[:, None] + 1e-12)), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        alpha_list_c.append(scale_tensor_all_c.clone())
        binary= torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # 3. 交替优化 (修改迭代公式)
    for k in range(iter):
        # 更新 alpha for order 0
        W_tilde = new_matrix - (alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
        alpha_c_B = alpha_list_c[0][None, :] * binary_list[0] * mask
        # 修改: 引入加权最小二乘
        weighted_alpha_c_B_sq = alpha_c_B.pow(2) * error_weights
        alpha_list_r[0] = torch.sum(alpha_c_B * W_tilde * error_weights, dim=1) / (torch.sum(weighted_alpha_c_B_sq, dim=1) + 1e-8)

        alpha_r_B =  alpha_list_r[0][:, None] * binary_list[0] * mask
        weighted_alpha_r_B_sq = alpha_r_B.pow(2) * error_weights
        alpha_list_c[0] = torch.sum(alpha_r_B * W_tilde * error_weights, dim=0) / (torch.sum(weighted_alpha_r_B_sq, dim=0) + 1e-8)

        # 更新 alpha for order 1
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0]) * mask
        alpha_c_B = alpha_list_c[1][None, :] * binary_list[1] * mask
        weighted_alpha_c_B_sq = alpha_c_B.pow(2) * error_weights
        alpha_list_r[1] = torch.sum(alpha_c_B * W_tilde * error_weights, dim=1) / (torch.sum(weighted_alpha_c_B_sq, dim=1) + 1e-8)

        alpha_r_B =  alpha_list_r[1][:, None] * binary_list[1] * mask
        weighted_alpha_r_B_sq = alpha_r_B.pow(2) * error_weights
        alpha_list_c[1] = torch.sum(alpha_r_B * W_tilde * error_weights, dim=0) / (torch.sum(weighted_alpha_r_B_sq, dim=0) + 1e-8)

        # 更新 B (二值矩阵)
        # 修改: 引入加权误差来寻找最近的二值组合
        new_matrix_expanded = new_matrix.unsqueeze(-1)
        comb0 = alpha_list_r[0].reshape(-1, 1) @ alpha_list_c[0].reshape(1, -1)
        comb1 = alpha_list_r[1].reshape(-1, 1) @ alpha_list_c[1].reshape(1, -1)
        v = torch.stack([-comb0 - comb1, -comb0 + comb1,
                    comb0 - comb1, comb0 + comb1], dim=2)

        # 误差项乘以权重，使得显著位置的误差在比较中更重要
        weighted_error = torch.abs(new_matrix_expanded - v) * error_weights.unsqueeze(-1)
        min_indices = torch.argmin(weighted_error, dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1

    # 4. Final refine results (与原函数相同)
    sum_order_alternating = (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] + alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order1_rc_nomean_hessian(x, mask, Hinv_diag, order=1, iter=15, lambda_salience=2.0):
    """
    Hessian-aware version of order-1 row-column alternating optimization.
    Prioritizes salient weights during optimization using weighted least squares.
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)

    # 1. 生成显著性掩码和加权矩阵
    mask_non_salient, mask_salient = saliency_mask_from_hessian(x, Hinv_diag)
    error_weights = torch.ones_like(x)
    error_weights[mask_salient] = lambda_salience

    # 2. 初始化参数 (与原函数相同)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1

    # 初始化过程保持不变
    residual = new_matrix - sum_order
    masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
    scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
    scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
    scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / (scale_tensor_all_r[:, None] + 1e-12)), dim=0)
    scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
    binary = torch.sign(masked_x_tensor)
    new_binary = binary.clone()
    sum_order = sum_order + (scale_tensor_all_r[:, None] * scale_tensor_all_c[None, :] * binary)*mask

    # 3. 交替优化 (修改为加权最小二乘)
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()
    for k in range(iter):
        # 3-1. 更新 alpha_r (行缩放)
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        # [MODIFIED] Numerator and denominator are weighted by error_weights
        num_r = torch.sum(alpha_c_B * new_matrix * error_weights, dim=1)
        den_r = torch.sum(alpha_c_B.pow(2) * error_weights, dim=1) + 1e-8
        new_alpha_r = num_r / den_r

        # 3-2. 更新 alpha_c (列缩放)
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        # [MODIFIED] Numerator and denominator are weighted by error_weights
        num_c = torch.sum(alpha_r_B * new_matrix * error_weights, dim=0)
        den_c = torch.sum(alpha_r_B.pow(2) * error_weights, dim=0) + 1e-8
        new_alpha_c = num_c / den_c

    # 4. Final refine results (与原函数相同)
    sum_order_alternating = new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask
    return sum_order_alternating

class Binarization(nn.Module):
    def __init__(self, weight, method="arb", groupsize=-1):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        self.method=method
        self.mean = 0

    def quantize(self, w, mask=None, Hinv_diag=None, order=2, groupi=0, S=None):
        if self.method=="xnor":
            w_mean = self.mean[groupi]
            w = w - w_mean  # oc, ic
            w = w.sign()
            w = w * self.scale[groupi]
            w+=w_mean
        elif self.method=="braq": # The method used in BiLLM
            w = high_order_residual(w, mask, order=order)

        # arb series
        elif self.method == "arb":
            if order == 2:
                w = high_order_residual_alternating_mean(w, mask, order=order)
            else:
                w = high_order_residual_alternating_order1(w, mask, order=order)
        elif self.method == 'arb-x':
            if order == 2:
                #w = high_order_residual_alternating_mean_x(w, mask, order=order, S=S)
                #w = high_order_residual_alternating_mean_x_new(w, mask, order=order, S=S)
                w = high_order_residual_alternating_mean_x_new_revised(w, mask, Hinv_diag, order=order, S=S, num_iters=15, lambda_salience=2.0)
            else:
                w = high_order_residual_alternating_order1_x(w, mask, order=order, S=S)
        elif self.method == 'arb-rc':
            if order == 3:
                #w = high_order_residual_alternating_order3_rc_nomean(w, mask, order=order)
                w = high_order_residual_alternating_order3_rc_nomean_hessian(w, mask, Hinv_diag, order=3, iter=15, lambda_salience=3.0)
            elif order == 2:
                #w = high_order_residual_alternating_order2_rc_nomean(w, mask, order=order)
                w = high_order_residual_alternating_order2_rc_nomean_hessian(w, mask, Hinv_diag, order=2, iter=15, lambda_salience=3.0)
            elif order == 1:
                #w = high_order_residual_alternating_order1_rc_nomean(w, mask, order=order)
                w = high_order_residual_alternating_order1_rc_nomean_hessian(w, mask, Hinv_diag, order=1, iter=15, lambda_salience=3.0)

        elif self.method=="sign":
            w=(w>0).float()
            w*=self.scale[groupi]
        elif self.method=="rtn":
            w=F.relu(w)
            w_int=(w/self.scale[groupi]).round().clamp(0,1)
            w=w_int*self.scale[groupi]
        elif self.method in ['2bit','4bit']:

            bits = int(self.method[0])
            perchannel = True
            weight = True
            dev = w.device
            maxq = torch.tensor(2 ** bits - 1)
            scale = torch.zeros(1)
            zero = torch.zeros(1)

            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)

            x = w.clone()
            shape = x.shape

            if perchannel:
                if weight:
                    x = x.flatten(1)
                else:
                    if len(shape) == 4:
                        x = x.permute([1, 0, 2, 3])
                        x = x.flatten(1)
                    if len(shape) == 3:
                        x = x.reshape((-1, shape[-1])).t()
                    if len(shape) == 2:
                        x = x.t()
            else:
                x = x.flatten().unsqueeze(0)
            tmp = torch.zeros(x.shape[0], device=dev)
            xmin = torch.minimum(x.min(1)[0], tmp)
            xmax = torch.maximum(x.max(1)[0], tmp)

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            if not perchannel:
                if weight:
                    tmp = shape[0]
                else:
                    tmp = shape[1] if len(shape) != 3 else shape[2]
                scale = scale.repeat(tmp)
                zero = zero.repeat(tmp)

            if weight:
                shape = [-1] + [1] * (len(shape) - 1)
                scale = scale.reshape(shape)
                zero = zero.reshape(shape)
            w = normal_quantize(w, scale, zero, maxq)

        elif self.method=="prune":
            return torch.zeros_like(w)
        return w
