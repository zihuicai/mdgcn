import torch
from torch import nn


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer



# Select the features of {idx} from {x}
# x--(batch_size, num_dims, num_points, 1), idx--(batch_size, num_points, k)
# return: (batch_size, num_dims, num_points, k)
def batched_index_select(x, idx):
    batch_size, num_dims, num_vertices_reduced = x.shape[ :3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)
    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature

# convolution，input--(batch_size, num_dims, num_points, 1)
class BasicConv(nn.Sequential):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop_out=0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[i]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop_out > 0:
                m.append(nn.Dropout2d(drop_out))
        super(BasicConv, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
class MRConv2d(nn.Module):
    def __init__(self, in_channels, act='relu', norm=None, bias=True, drop_out=0.0):
        super(MRConv2d, self).__init__()
        self.mlp = BasicConv([2*in_channels, in_channels], act, norm, bias, drop_out)

    # x--(batch_size, num_dims, num_points, 1)
    # edge_index--(2, batch_size, num_points, k), represents the k neighbors
    def forward(self, x, edge_index):
        # edge_index--(batch_size, num_points, k)，x_i--(batch_size, num_dims, num_points, k)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, dim=-1, keepdim=True)
        batch_size, num_dims, num_points, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2)
        x = x.reshape(batch_size, 2 * num_dims, num_points, _)
        # x--(batch_size, 2*num_dims, num_points, 1) => (batch_size, num_dims, num_points, 1)
        x = self.mlp(x)
        return x


# mask the features of padding nodes, set them to zeros
# x--(batch_size, num_dims, num_points, 1) mask--(batch_size, num_points)
def mask_isolatedVertex(x, mask):
    batch_size, num_dims, num_points, _ = x.shape
    mask = mask.unsqueeze(dim=1).expand(batch_size, num_dims, num_points)
    mask = mask.unsqueeze(dim=-1)
    x = x.masked_fill_(mask, 0.0)
    return x

# ignoring the padding nodes when meaning
# x--(batch_size, num_dims, num_points, 1) mask--(batch_size, num_points)
def valid_mean(x, mask):
    # x--(batch_size, num_dims, num_points)
    x = x.squeeze(-1)
    # x_mean--(batch_size, num_dims)
    x_mean = torch.zeros(x.shape[: 2], dtype=x.dtype, device=x.device)
    # process each sample
    batch_size, num_points = mask.shape
    for i in range(batch_size):
        # x[i]--(num_dims, num_points) mask[i]--(num_points, )
        num_effective = num_points - torch.sum(mask[i])
        mean_i = torch.sum(x[i, :, :num_effective], dim=-1) / num_effective
        x_mean[i] = mean_i
    x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
    return x_mean



