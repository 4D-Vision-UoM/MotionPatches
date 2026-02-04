"""Definition of Point Encoder"""

# Credits: https://github.com/theamaya/CrossMoST
# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name


import torch
from torch import nn



def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return index_points(xyz, centroids)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


# class TransformerEncoder(nn.Module):
#     """Transformer Encoder without hierarchical structure"""

#     # pylint: disable=too-many-positional-arguments
#     def __init__(
#         self,
#         embed_dim=768,
#         depth=4,
#         num_heads=12,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.0,
#     ):
#         super().__init__()

#         # pylint: disable=duplicate-code
#         self.blocks = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dim,
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate,
#                     drop_path=(
#                         drop_path_rate[i]
#                         if isinstance(drop_path_rate, list)
#                         else drop_path_rate
#                     ),
#                 )
#                 for i in range(depth)
#             ]
#         )
#         # pylint: enable=duplicate-code

#     def forward(self, x, pos):
#         for _, block in enumerate(self.blocks):
#             x = block(x + pos)
#         return x


class PointnetEncoder(nn.Module):
    def __init__(self, num_input=2048, num_groups=64, group_size=32, encoder_dim=512, return_centroids=False):
        super().__init__()
        self.num_input = num_input
        self.num_groups = num_groups
        self.group_size = group_size
        self.encoder_dim = encoder_dim
        self.return_centroids = return_centroids
        
        self.encoder = Encoder(encoder_channel=self.encoder_dim)
        self.group_divider = Group(num_group=self.num_groups, group_size=self.group_size)

    def forward(self, x):
        """
        x: B N 3
        ----------------
        feature_global: B G C (B, 64, 512) -> can be reshaped to (B, 512, 8, 8) for ViT
        """
        neighborhood, centroids = self.group_divider(x)  # B G M 3
        feature_global = self.encoder(neighborhood)  # B G C
        # feature_global = torch.max(feature_global, dim=1, keepdim=False)[0]  # B C
        if self.return_centroids:
            return feature_global, centroids
        return feature_global


