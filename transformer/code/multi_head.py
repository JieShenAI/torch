import copy

import torch
from torch import nn
import math


# 定义通用的Attention函数
def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        :param h: number of heads.
        :param d_model:
        :param dropout:
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # 每个头的维度
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attn = None
        # self.dropout = nn.Dropout(p=dropout)

    # def forward(self, query, key, value, mask=None):
    def forward(self, x):
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        # nbatches = query.size(0)

        nbatches = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [
        #     l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #     for l, x in zip(self.linears, (query, key, value))
        # ]

        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (x, x, x))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = attention(query, key, value, mask=mask,
        #                          dropout=self.dropout)
        x = attention(query, key, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2) \
            .contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


if __name__ == '__main__':
    model = MultiHeadedAttention(8, 512)
    x = torch.randn(2, 7, 512)
    print(model(x).size())
