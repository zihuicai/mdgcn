from src.mdgcn.components import MRConv2d, act_layer, mask_isolatedVertex, valid_mean
from src.mdgcn.knn import knn
import torch
from torch import nn
from timm.models.layers import DropPath


# construct graph once, and perform convolution multiple times
class MDyGraphConv2d(nn.Module):
    def __init__(self, in_channels, conv_times, inner_k, cross_k,
                 act, norm, drop_out=0.0):
        super(MDyGraphConv2d, self).__init__()
        self.inner_k = inner_k
        self.cross_k = cross_k
        self.conv_layers = nn.ModuleList()
        for _ in range(conv_times):
            conv_layer = MRConv2d(in_channels, act, norm, drop_out=drop_out)
            self.conv_layers.append(conv_layer)

    # x--(batch_size, num_dims, x_num_points, 1) y--(batch_size, num_dims, y_num_points, 1)
    def forward(self, x, y):
        # (2, batch_size, num_points, k or k+3)
        edge_index = knn(x, y, inner_k=self.inner_k, cross_k=self.cross_k)
        # graph convolution and residual connection
        xy = torch.cat([x, y], dim=-2)
        for conv_layer in self.conv_layers:
            xy = conv_layer(xy, edge_index) + xy
        # separate
        x_num_points = x.size(-2)
        x = xy[:, :, : x_num_points]
        y = xy[:, :, x_num_points:]
        return x, y

# Multimodal Grapher
class MGrapher(nn.Module):
    def __init__(self, in_channels, conv_times, inner_k, cross_k,
                 act, norm, drop_out=0.0, drop_path=0.0):
        super(MGrapher, self).__init__()
        self.graph_conv = MDyGraphConv2d(in_channels, conv_times, inner_k, cross_k,
                                         act, norm, drop_out)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        # drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y):
        x_residual, y_residual = x, y
        x, y = self.graph_conv(x, y)
        x = self.fc(x)
        y = self.fc(y)
        x = self.drop_path(x) + x_residual
        y = self.drop_path(y) + y_residual
        return x, y

# Feed Forward
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act='gelu', drop_out=0.0, drop_path=0.0):
        super(FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
            act_layer(act),
            nn.Dropout2d(drop_out)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
            act_layer(act),
            nn.Dropout2d(drop_out)
        )
        # drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop_path(x) + x_residual
        return x

# MDGC Encoder Layer：mgrapher + ffn
class MDGCNBlock(nn.Module):
    def __init__(self, channels, conv_times, inner_k, cross_k, drop_out=0.0, drop_path=0.0):
        super(MDGCNBlock, self).__init__()
        self.mgrapher = MGrapher(channels, conv_times, inner_k, cross_k, act='gelu',
                                 norm='batch', drop_out=drop_out, drop_path=drop_path)
        self.ffn = FFN(channels, 4*channels, act='gelu', drop_out=drop_out, drop_path=drop_path)

    def forward(self, x, x_mask, y, y_mask):
        x, y = self.mgrapher(x, y)
        x = self.ffn(x)
        y = self.ffn(y)
        # mask the padding nodes
        x = mask_isolatedVertex(x, x_mask)
        y = mask_isolatedVertex(y, y_mask)
        return x, y

# MDGCN
class MDGCN(nn.Module):
    def __init__(self, channels, conv_times_list, inner_k_list, cross_k_list, drop_out=0.0, drop_path=0.0):
        super(MDGCN, self).__init__()
        block_num = len(conv_times_list)
        assert (len(inner_k_list) == block_num) and (len(cross_k_list) == block_num)
        # Multiple MDGC Encoder Layers
        self.mdgcn_blocks = nn.ModuleList()
        for i in range(block_num):
            block = MDGCNBlock(channels, conv_times_list[i], inner_k_list[i],
                               cross_k_list[i], drop_out, drop_path)
            self.mdgcn_blocks.append(block)
        # classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(2*channels, channels//2, 1, bias=True),
            nn.BatchNorm2d(channels//2),
            act_layer('gelu'),
            nn.Dropout2d(drop_out),
            nn.Conv2d(channels//2, 2, 1, bias=True),
            nn.Flatten()
        )

    def forward(self, inputs):
        # x--(batch_size, num_points, num_dims)
        x_inputs, y_inputs = tuple(inputs)
        x, x_mask = tuple(x_inputs)
        y, y_mask = tuple(y_inputs)
        # reshape x and y, x--(batch_size, num_dims, num_points, 1)
        x = x.transpose(1, 2).unsqueeze(-1)
        y = y.transpose(1, 2).unsqueeze(-1)
        # multiple layers
        for block in self.mdgcn_blocks:
            x, y = block(x, x_mask, y, y_mask)
        # avg_x--(batch_size, num_dims, 1, 1)
        avg_x = valid_mean(x, x_mask)
        avg_y = valid_mean(y, y_mask)
        avg_xy = torch.cat([avg_x, avg_y], dim=1)
        # classify
        out = self.classifier(avg_xy)
        return out



