import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([1.0, 2.0])
true_b = 1.0


def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise.

    Defined in :numref:`sec_linear_scratch`"""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    # y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))


features, labels = synthetic_data(true_w, true_b, 5)
print(features, labels)
print(labels.shape, labels.size())

