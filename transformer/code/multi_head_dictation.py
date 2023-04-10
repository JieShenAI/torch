import torch
from torch import nn
import math
import copy


# 定义通用的注意力函数
def attention(query, key, value):
    d_k = query.size(-1)
    k_T = key.transpose(-2, -1)
    qk = torch.matmul(query, k_T) / d_k
    qk = qk.softmax(dim=-1)
    return torch.matmul(qk, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.h_d = d_model // h  # 每个头的维度数
        self.linear = nn.Linear(self.h_d, self.h_d, bias=False)

    def forward(self, x):
        # 拆分出多个头
        batch = x.size(0)
        x = x.view(batch, -1, self.h, self.h_d)
        # 把头的维度换到第二个维度
        x = x.transpose(1, 2)
        query, key, value = [
            linear(data)
            for linear, data in zip(clones(self.linear, 3), (x, x, x))
        ]
        y = attention(query, key, value)
        return y.transpose(1, 2).contiguous().view(batch, -1, self.h * self.h_d)


def clones(module, cnt):
    return nn.ModuleList([
        copy.deepcopy(module)
        for i in range(cnt)
    ])


if __name__ == '__main__':
    x = torch.randn(32, 7, 256)
    model = MultiHeadAttention(4, 256)
    res = model(x)
    print(res.shape)
