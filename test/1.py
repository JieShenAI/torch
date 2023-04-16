import torch
from torch import nn
import pandas as pd


# a = torch.tensor([3, 4])
# b = torch.tensor([3, 4])
# c = torch.tensor([3, 4])
# print(a.shape)
# t = torch.concat((a, b, c), dim=1)
# print(t, t.shape)


def kaggle_submissions(example_csv, data, submission_csv='submission.csv'):
    """
    给定一行可迭代的对象，添加到csv指定的列中
    :return:
    """
    df = pd.read_csv(example_csv)
    print(len(df))


if __name__ == '__main__':
    # base = "D:\\github\\torch\\huggingface\\example\\bbc_news_classification\\data\\"
    # submit = "BBC News Sample Solution.csv"
    # kaggle_submissions(base + submit)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.tensor([
        [99., 2., 3.],
        [4., 5., 6.],
        [4., 55., 36.]
    ], dtype=torch.float32)

    a = torch.tensor(a)
    print(a)
    print(a.to(device))
    print(a.argmax(-1))
