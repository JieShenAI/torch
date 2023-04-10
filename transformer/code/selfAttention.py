import torch
from torch import nn
import torch.nn.functional as F
import math


class selfAttention(nn.Module):
    def __init__(self,
                 input_vector_dim: int,
                 dim_k_hidden=None,
                 dim_v=None,
                 ):
        super(selfAttention, self).__init__()
        self.input_vector_dim = input_vector_dim
        if dim_k_hidden is None:
            dim_k_hidden = input_vector_dim
        if dim_v is None:
            dim_v = input_vector_dim

        self.W_q = nn.Linear(input_vector_dim, dim_k_hidden, bias=False)
        self.W_k = nn.Linear(input_vector_dim, dim_k_hidden, bias=False)
        self.W_v = nn.Linear(input_vector_dim, dim_v, bias=False)
        self._norm_factor = 1 / math.sqrt(dim_k_hidden)

    def forward(self, x):
        Q = self.W_q(x)  # d, dim_k_hidden
        K = self.W_k(x)  # d, dim_k_hidden
        V = self.W_v(x)  # d, dim_v

        # 注意这个地方，是对K进行转置
        k_T = K.transpose(1, 2)  # dim_k_hidden, d

        # d, d
        attention = nn.Softmax(dim=-1)(torch.bmm(Q, k_T) * self._norm_factor)
        output = torch.bmm(attention, V)  # d, dim_v
        return output


if __name__ == '__main__':
    n = 128  # 最长的句子长度
    dim_k = 12
    dim_v = 43
    attention = selfAttention(n, )
    # output = attention(torch.Tensor([50, 5, 3]))
    # 输入的shape是 d x n
    batch_size = 32
    d = 314  # 词向量的维度

    output = attention(torch.randn(batch_size, d, n))
    print(output.shape)  # 32, 43, 128
