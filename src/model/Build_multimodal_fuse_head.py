import torch
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import to_2tuple
from torch import nn
from einops.layers.torch import Rearrange
import numpy as np
import functools
import pandas as pd


def creat_norm_layer(norm_layer, channel, is_token=False):
    """
    Args:
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'.
        channel (int): Input channels.
        is_token (bool): Whether to process token. Default: False
    """
    if not is_token:
        if norm_layer == 'LN':
            norm = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(channel),
                Rearrange('b h w c -> b c h w')
            )
        elif norm_layer == 'BN':
            norm = nn.BatchNorm2d(channel)
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    else:
        if norm_layer == 'LN':
            norm = nn.LayerNorm(channel)
        elif norm_layer == 'BN':
            norm = nn.Sequential(
                Rearrange('b d n -> b n d'),
                nn.BatchNorm1d(channel)
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


class MSE(nn.Module):
    """
    Build MSE

    Args:
        in_chans (int): Number of input image channels. Default: 3
        out_chans (int): Number of output image channels. Default: 24

    Return shape: (b c H W)
    """

    def __init__(self, in_chans, out_chans, n_group=4, use_pos=True, channel_attn_type='SE', ratio=16):
        super().__init__()
        self.use_pos = use_pos

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(out_chans, out_chans // 2, kernel_size=1, bias=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_chans // 2, out_chans // 2, kernel_size=3, padding=1, groups=n_group),
            nn.BatchNorm2d(out_chans // 2),
            nn.Conv2d(out_chans // 2, out_chans, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        if channel_attn_type == 'SE':
            self.attn = SE_channel_attention(out_chans, ratio)
        else:
            self.attn = CBAM_channel_attention(out_chans, ratio)

    def forward(self, x, pos):
        x = self.conv1(x)
        short_cut = x
        x = self.conv2(x)
        if self.use_pos:
            b, c, H, W = x.shape
            pos = repeat(pos, '1 -> b c H W', b=b, c=c, H=H, W=W)
            x = x + pos
        x = self.conv3(x)
        x = x + short_cut
        x = self.attn(x)

        return x


class AMM(nn.Module):
    """
    Creat AMM module

    Args:
        in_chans (int): Number of input image channels.
        out_chans (int): Number of output image channels.
        n_heads (int): Number of attention heads. Default: 4
        n_branch (int): Number of branches.
        patch_size (int | tuple[int]): Patch size. Default: 4
        n_heads (int): Number of attention heads.
        fuse_drop (float): Dropout rate.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True

    Return shape: (b c h w)
    """

    def __init__(self, in_chans,
                 out_chans,
                 n_branch,
                 offset_scale=16,
                 patch_size=4,
                 n_heads=4,
                 fuse_drop=0.,
                 qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.patch_size = to_2tuple(patch_size)

        self.short_cut_conv = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size),
                                            creat_norm_layer('LN', out_chans))

        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.q_proj = nn.Sequential(nn.MaxPool2d(offset_scale, stride=offset_scale),
                                    nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, groups=in_chans))
        self.k_proj = nn.Sequential(nn.MaxPool2d(offset_scale, stride=offset_scale),
                                    nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, groups=in_chans))
        self.v_proj = nn.Conv2d(in_chans, in_chans, kernel_size=patch_size, stride=patch_size, groups=in_chans)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(1, 16 * n_branch, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(16 * n_branch, n_heads, bias=False))

        coords = torch.zeros([in_chans, in_chans], dtype=torch.int64)
        for idx in range(in_chans):
            coords[idx] = torch.arange(in_chans) - idx
        relative_position_bias = coords / coords.max()
        relative_position_bias *= 8  # normalize to -8, 8
        relative_position_bias = torch.sign(relative_position_bias) * torch.log2(torch.abs(relative_position_bias) + 1.0) / np.log2(8)
        self.register_buffer("relative_position_bias", relative_position_bias.unsqueeze(-1))

        self.dropout = nn.Dropout(fuse_drop)
        self.norm = creat_norm_layer('LN', out_chans)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax1 = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(nn.Conv2d(in_chans, in_chans, kernel_size=1),
                                  nn.GELU(),
                                  nn.Conv2d(in_chans, out_chans, kernel_size=1))

    def forward(self, x):
        short_cut = x
        b, c, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)  # b, c, h, w
        q, k, v = self.q_proj(q).flatten(2), self.k_proj(k).flatten(2), self.v_proj(v).flatten(2)  # b, c, h*w
        q = q.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)  # b, n, c, h*w//n
        k = k.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)

        # cosine attention
        sim = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(sim.device))).exp()
        sim = sim * logit_scale

        relative_position_bias = self.cpb_mlp(self.relative_position_bias).view(-1, self.n_heads).view(c, c, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = torch.sigmoid(relative_position_bias)
        sim = sim + relative_position_bias.unsqueeze(0)

        sim = self.softmax1(1 - self.softmax(sim))
        sim = self.dropout(sim)
        x = (sim @ v).transpose(1, 2).reshape(b, c, -1)  # b, c, h*w
        x = x.view(b, -1, H // self.patch_size[0], W // self.patch_size[1])  # b, c, h, w
        x = self.proj(x)
        x = self.dropout(x)
        x = self.norm(x) + self.short_cut_conv(short_cut)

        return x, short_cut


class SE_channel_attention(nn.Module):
    """
    Build channel attention based on SE

    Args:
        in_chans (int): Number of input image channels.
        ratio (int): Scaling ratio. Default: 4
        act_layer (nn.Module): Act layer. Default: nn.ReLU6

    Return shape: (b c h w)
    """

    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class CBAM_channel_attention(nn.Module):
    """
    Build channel attention based on CBAM

    Args:
        in_chans (int): Number of input image channels.
        ratio (int): Scaling ratio. Default: 4

    Return shape: (b c h w)
    """

    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.conv2 = nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.act_layer(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.act_layer(self.conv1(self.max_pool(x))))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight

class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4, num_heads=8):
        super().__init__()
        self.num_modals = num_modals
        # self.num_heads = num_heads
        # self.embed_dim = embed_dim

        # self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)

        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        ) for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_attn = []
        # for i in range(self.num_modals):
        #     x_reshape = x[i].view(B, C, -1).permute(2, 0, 1)  # Reshape for multi-head attention
        #     attn_output, _ = self.attention(x_reshape, x_reshape, x_reshape)
        #     attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)  # Reshape back
        #     x_attn.append(attn_output)

        # for i in range(self.num_modals):
        #     x[i] = x[i].view(B, C, -1).permute(2, 0, 1)  # Reshape for multi-head attention
        # x_cat = torch.cat(x, dim=0)
        # attn_output, _ = self.attention(x_cat, x_cat, x_cat)
        # attn_output = attn_output.permute(1, 2, 0)
        # for i in range(self.num_modals):
        #     x_attn.append(attn_output[:, :, i*(H*W):(i+1)*(H*W)].view(B, C, H, W))
        #     x[i] = x[i].permute(1, 2, 0).view(B, C, H, W)

        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            # x_attn[i] = self.norm1(x_attn[i].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Normalization
            # x_[i] = self.score_nets[i](x_attn[i])
            # x_[i] = self.norm2(x_[i])  # Normalization
            x_[i] = self.score_nets[i](x[i])

        return x_

    def tokenselect(self, x_ext, module):
        x_scores = module(x_ext)

        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]

        score_P = torch.sum(x_ext[0] > x_ext[1]).item() / (x_ext[0].shape[0] * x_ext[0].shape[1] * x_ext[0].shape[2] * x_ext[0].shape[3])

        P = pd.DataFrame({score_P})
        P.to_csv('/home/dell/Project/IHNet/runs/score_P_noColor.csv',mode='a',index=False,index_label=None,  header=False)
        # np.savetxt('/home/dell/Project/IHNet/runs/score_P.txt',str(score_P)+"% ")

        x_f = functools.reduce(torch.max, x_ext)

        # return x_f, x_scores
        return x_f


class Build_multimodal_fuse_head(nn.Module):
    """
    Build multimodal_fuse_head

    Args:
        n_branch (int): Number of branches.
        in_chans (int, tuple[int]): Number of channels for input images in each branch. Default: (3, 3, 3, 3)
        out_chans (int): Number of output image channels. Default: 36
        n_group (int): Number of groups.
        patch_size (int): Branch self-attention patch size. Default: 4
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        chan_ratio (int): Scaling ratio of 'CBAM' or 'SE' channel attention. Default: 16
        n_heads (int): Number of SA channel attention heads. Default: 4
        fuse_type: AMM method.

    Return shape: (b c h w)
    """

    def __init__(self,
                 n_branch,
                 in_chans=(3, 3, 3, 3),
                 out_chans=36,
                 n_group=3,
                 use_pos=True,
                 patch_size=4,
                 attn_drop=0.1,
                 qkv_bias=False,
                 offset_scale=8,
                 chan_ratio=16,
                 chan_attn_type='SE',
                 n_heads=2,
                 fuse_type=None,
                 embed_dim=None):
        super().__init__()
        in_chans = in_chans if isinstance(in_chans, tuple) else tuple([in_chans for _ in range(n_branch)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fuse_type = fuse_type
        self.use_pos = use_pos

        self.score = PredictorConv(out_chans, 2)

        self.MSEs = nn.ModuleList([
            MSE(in_chans=in_chans[i],
                out_chans=out_chans,
                n_group=n_group,
                use_pos=use_pos,
                channel_attn_type=chan_attn_type,
                ratio=chan_ratio)
            for i in range(n_branch)])

        if use_pos:
            ang_table = [ang for ang in range(0, 136, 135 // n_branch)]
            self.pos = [nn.Parameter(torch.tensor([np.cos(ang_table[i] * np.pi / 180)], dtype=torch.float32))
                        for i in range(n_branch)]

        # smooth_chans = int(out_chans * n_branch)

        smooth_chans = int(out_chans)

        self.smooth = nn.Sequential(
            nn.Conv2d(smooth_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
        )

        if self.fuse_type is None:
            self.fuse_proj = AMM(in_chans=smooth_chans,
                                 out_chans=embed_dim,
                                 n_branch=n_branch,
                                 n_heads=n_heads,
                                 offset_scale=offset_scale,
                                 patch_size=patch_size,
                                 fuse_drop=attn_drop,
                                 qkv_bias=qkv_bias)
        else:
            self.fuse_proj = nn.Identity()

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        fuse = []
        for i, layer in enumerate(self.MSEs):
            x_branch = layer(x[i], self.pos[i].cuda(non_blocking=True) if self.use_pos else None)
            # x_branch = layer(x[i], self.pos[i].to(torch.device('cuda:1')) if self.use_pos else None)
            fuse.append(x_branch)

        # x = self.fuse_proj(torch.cat(fuse, dim=1))
        # if self.fuse_type is not None:
        #     x = self.smooth(x)
        #     return x
        # else:
        #     de_x = self.smooth(x[1])
        #     return x[0], de_x

        # x, x_scores = self.score.tokenselect(fuse, self.score)
        x = self.score.tokenselect(fuse, self.score)
        x = self.smooth(x)
        # return x, x_scores
        return x
