import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.net_utils import DropPath, get_pad_mask
from einops import rearrange


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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attention = ScaledDotProductAttention(temperature=qk_scale or head_dim ** 0.5)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, mask=None):
        B, Nq, Nk, Nv, C = q.shape[0], q.shape[1], k.shape[1], v.shape[1], q.shape[2]
        if self.with_qkv:
            q = self.proj_q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = self.proj_k(k).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = self.proj_v(v).reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        else:
            q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attn = self.attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).reshape(B, Nq, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, N, mask=None):

        if mask is not None:
            src_mask = rearrange(mask, 'b n t -> b (n t)', b=B, n=N, t=T)
            src_mask = get_pad_mask(src_mask, 0)
        else:
            src_mask = None
        x2 = self.norm1(x)
        x = x + self.drop_path(self.attn(q=x2, k=x2, v=x2, mask=src_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.enc_dec_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, tgt, memory, memory_mask=None, trg_mask=None):
        tgt_2 = self.norm1(tgt)
        tgt = tgt + self.drop_path(self.self_attn(q=tgt_2, k=tgt_2, v=tgt_2, mask=trg_mask))
        tgt = tgt + self.drop_path(self.enc_dec_attn(q=self.norm2(tgt), k=memory, v=memory, mask=memory_mask))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        return tgt
