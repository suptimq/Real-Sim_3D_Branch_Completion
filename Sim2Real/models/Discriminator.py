from .build import MODELS

import torch
import torch.nn as nn

from timm.models.layers import DropPath,trunc_normal_


class KNNAttention(nn.Module):
    # TODO add normalization and residule
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, K, C = x.shape
        
        # 0, 1, 2, 3, 4, 5 ->  3, 0, 4, 1, 2, 5
        # B, N, K, 3, H, C' -> 3, B, H, N, K, C'
        qkv = self.qkv(x).reshape(B, N, K, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        # B, H, N, K, C'
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # B, H, N, K, K
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 0, 1, 2, 3, 4 ->  0, 2, 3, 1, 4
        # B, H, N, K, C' -> B, N, K, H, C'
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, N, K, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = KNNAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Args:
            x: B, N, K, C
        """
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)
        
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODELS.register_module()
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.trans_dim = config.trans_dim
        self.dropout = config.dropout
        self.attn_drop = config.attn_drop

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 128, 1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, self.trans_dim, 1)
        )

        self.attn_block = Block(dim=self.trans_dim*2,
                                num_heads=6,
                                mlp_ratio=2.,
                                qkv_bias=False,
                                qk_scale=None,
                                drop=self.attn_drop,
                                drop_path=self.attn_drop)

        self.second_conv = nn.Sequential(
            nn.Conv2d(self.trans_dim*2, 512, 1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, 1)
        )

        self.fc = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 1, 1)
        )

    def forward(self, xyz):
        """
        Args:
            xyz: input points, bs, 3, N, K
        """

        bs, _, _, K = xyz.shape
        feature = self.first_conv(xyz)  # B 256 N K
        feature_global = torch.max(feature, dim=-1, keepdim=True)[0]  # B 256 N 1
        feature = torch.cat([feature_global.expand(-1, -1, -1, K), feature], dim=1) # B 512 N K

        feature = self.attn_block(feature.permute(0, 2, 3, 1))   # B N K 512
        feature = self.second_conv(feature.permute(0, 3, 1, 2))  # B 512 N K
        feature_global = torch.max(feature, dim=-1, keepdim=False)[0]  # B 512 N

        outputs = self.fc(feature_global).view(bs, -1)   # B N

        return outputs