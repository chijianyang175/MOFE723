'''
Author: Sicen Liu
Date: 2025-05-16 14:38:17
LastEditTime: 2025-06-18 20:08:48
FilePath: /liusicen/shared_files/MOFE/src/feature_level/modules.py
Description: modules

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        # x 的形状为 (batch_size, time_steps, ...)
        batch_size, time_steps, *feature_size = x.size()
        # 将输入数据重塑为 (batch_size * time_steps, ...)
        x = x.view(batch_size * time_steps, *feature_size)
        # 应用层
        x = self.layer(x)
        # 将输出重塑回 (batch_size, time_steps, ...)
        output_size = x.size()[1:]  # 获取输出的形状
        x = x.view(batch_size, time_steps, *output_size)
        return x

# 创建包含 TimeDistributed 和 Dense 层的模型
class TDModule(nn.Module):
    def __init__(self, in_f=512, out_f=512):
        super(TDModule, self).__init__()
        self.time_distributed = TimeDistributed(nn.Linear(in_features=in_f, out_features=out_f))  # 512 单元，激活函数在前向中应用

    def forward(self, x):
        x = self.time_distributed(x)
        x = torch.tanh(x)  # 应用 tanh 激活函数
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]  # batch size
        length = x.shape[1]
        
        # Split the embedding into multiple heads
        values = self.values(x).view(N, length, self.heads, self.head_dim)
        keys = self.keys(x).view(N, length, self.heads, self.head_dim)
        queries = self.queries(x).view(N, length, self.heads, self.head_dim)

        values, keys, queries = (
            values.permute(0, 2, 1, 3),
            keys.permute(0, 2, 1, 3),
            queries.permute(0, 2, 1, 3),
        )

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(
            N, length, self.heads * self.head_dim
        )

        return self.fc_out(out)


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, depth, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, heads, dropout) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

