import torch
import math


""""Get edges based on the similarity"""
# compute the similarity between x and y, some elements of {sim} may be nan
# x--(batch_size, x_num_points, num_dims) y--(batch_size, y_num_points, num_dims)
def xy_pairwise_sim(x, y):
    with torch.no_grad():
        xy_inner = torch.bmm(x, y.transpose(-1, -2))
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        # cosine similarity, (batch_size, x_num_points, y_num_points)
        sim = xy_inner / torch.bmm(x_norm, y_norm.transpose(-1, -2))
        return sim

# get top k elements, cycle padding when {ele_num} < k
def get_topk(tensor, k):
    assert len(tensor.shape) == 3
    ele_num = tensor.size(-1)
    if ele_num >= k:
        _, res = torch.topk(tensor, k, dim=-1)
        return res
    # ele_num < k
    _, base_res = torch.topk(tensor, ele_num, dim=-1)
    repeat_rate = k // ele_num
    res = base_res.repeat(1, 1, repeat_rate)
    res = torch.cat([res, base_res[:, :, :k - repeat_rate * ele_num]], dim=-1)
    return res

# get the most similar nodes, pos_nn_idx--(batch_size, x_num_points, k)
def get_nn_idx_by_sim(pos_sim, k, y_offset):
    pos_nn_idx = get_topk(pos_sim, k)
    pos_nn_idx += y_offset
    return pos_nn_idx



"""Get edges based on the location"""
# get the closest nodes, adj_nn_idx--(batch_size, x_num_points, 3)
# inv_mask.shape equals to sim.shape, both of them are (batch_size, x_num_points, y_num_points)
def get_nn_idx_by_loc(inv_mask, x_offset, y_offset):
    batch_size, x_num_points, y_num_points = inv_mask.shape
    # get the effective node nums according to {inv_mask}
    x_effective_num_points_list = torch.sum(inv_mask[:, :, 0], dim=-1)
    y_effective_num_points_list = torch.sum(inv_mask[:, 0, :], dim=-1)
    # init post, self and pre to itself, adj_nn_idx--(batch_size, x_num_points, 3)
    adj_nn_idx = torch.arange(0, x_num_points, device=inv_mask.device)
    adj_nn_idx = adj_nn_idx.repeat(batch_size, 3, 1).transpose(2, 1)
    adj_nn_idx += x_offset
    # modify {adj_nn_idx}
    for i in range(batch_size):
        x_effective_num_points = x_effective_num_points_list[i]
        y_effective_num_points = y_effective_num_points_list[i]
        self = torch.arange(0, x_effective_num_points, device=inv_mask.device)
        if x_effective_num_points > y_effective_num_points:
            gap = math.ceil(x_effective_num_points / y_effective_num_points)
            self = torch.div(self, gap, rounding_mode='trunc')
        else:
            gap = torch.div(y_effective_num_points, x_effective_num_points, rounding_mode='floor')
            self = self * gap
        pre = (self + y_effective_num_points - 1) % y_effective_num_points
        post = (self + 1) % y_effective_num_points
        cell_adj_nn_idx = torch.stack((pre, self, post)).transpose(0, 1)
        cell_adj_nn_idx += y_offset
        adj_nn_idx[i, :x_effective_num_points] = cell_adj_nn_idx
    return adj_nn_idx



"""construct edges based on the similarity and location"""
# sim--(batch_size, x_num_points, y_num_points)
def construct_edges(sim, k, x_offset, y_offset):
    batch_size, x_num_points, y_num_points = sim.shape
    # k most similar edges and pre, self, post
    with torch.no_grad():
        # the most similar edges, pos_nn_idx--(batch_size, x_num_points, k)
        pos_sim = torch.where(torch.isnan(sim), -1.0, sim)
        pos_nn_idx = get_nn_idx_by_sim(pos_sim, k, y_offset)
        # the closest edges, adj_nn_idx--(batch_size, x_num_points, 3)
        inv_mask = torch.where(torch.isnan(sim), False, True)
        adj_nn_idx = get_nn_idx_by_loc(inv_mask, x_offset, y_offset)
        # concat, get k+3 edges
        nn_idx = torch.cat([pos_nn_idx, adj_nn_idx], dim=-1)
        # construct self, center_idx--(batch_size, x_num_points, k)
        center_idx = torch.arange(x_offset, x_offset + x_num_points, device=sim.device)
        center_idx = center_idx.repeat(batch_size, k + 3, 1).transpose(2, 1)
        # (2, batch_size, x_num_points, k)
        return torch.stack((nn_idx, center_idx), dim=0)

# select all edges between x and y
# x--(batch_size, num_dims, x_num_points, 1) y--(batch_size, num_dims, y_num_points, 1)
def knn(x, y, inner_k, cross_k):
    with torch.no_grad():
        # reshape x and y to (batch_size, num_points, num_dims)
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        x_num_points, y_num_points = x.size(1), y.size(1)
        # x and x
        xx_sim = xy_pairwise_sim(x, x)
        xx_edges = construct_edges(xx_sim, inner_k, x_offset=0, y_offset=0)
        # y and y
        yy_sim = xy_pairwise_sim(y, y)
        yy_edges = construct_edges(yy_sim, inner_k, x_offset=x_num_points, y_offset=x_num_points)
        # x and y
        xy_sim = xy_pairwise_sim(x, y)
        xy_edges = construct_edges(xy_sim, cross_k, x_offset=0, y_offset=x_num_points)
        # y and x
        yx_sim = xy_sim.transpose(-1, -2)
        yx_edges = construct_edges(yx_sim, cross_k, x_offset=x_num_points, y_offset=0)
        # combine
        x_edges = torch.cat([xx_edges, xy_edges], dim=-1)
        y_edges = torch.cat([yy_edges, yx_edges], dim=-1)
        edges = torch.cat([x_edges, y_edges], dim=-2)
        # (2, batch_size, x_num_points + y_num_points, edge_num)
        return edges






