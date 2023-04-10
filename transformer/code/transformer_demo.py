import torch
from torch import nn
import random
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0).to(device)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class CopyTaskModel(nn.Module):
    """
        src = torch.LongTensor([[0, 3, 4, 5, 6, 1, 2, 2]])
        tgt = torch.LongTensor([[3, 4, 5, 6, 1, 2, 2]])
        model = CopyTaskModel()
        out = model(src, tgt)
        print(out.size())
    """

    def __init__(self, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(10, 128, padding_idx=2).to(device)
        self.transformer = nn.Transformer(d_model, batch_first=True)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(128, 10)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        # print(tgt_mask.dtype)
        tgt_mask = tgt_mask.to(device)
        src_key_padding_mask = CopyTaskModel.get_key_padding_mask(src).to(device)
        tgt_key_padding_mask = CopyTaskModel.get_key_padding_mask(tgt).to(device)
        src = self.embedding(src)
        # 删去tgt的最后一个元素
        tgt = self.embedding(tgt)

        # 给src和tgt的token增加位置信息
        src_embedding = self.positional_encoding(src).to(device)
        tgt_embedding = self.positional_encoding(tgt).to(device)

        out = self.transformer(src_embedding,
                               tgt_embedding,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask
                               )
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        若token = padding，maks为 -torch.inf
        :param tokens:
        :return:

        eg:
            a = torch.Tensor([1, 3, 4, 5, 2, 2, 2])
            mask = CopyTaskModel.get_key_padding_mask(a)
        """
        mask = torch.zeros(tokens.size())
        mask[tokens == 2] = -torch.inf
        return mask


def generate_random_batch(batch_size, max_length=16):
    """

    :param batch_size:
    :param max_length:
    :return:

    eg:
        ans = generate_random_batch(2)
        for a in ans:
            print(a)
    """
    src = []
    for i in range(batch_size):
        random_num = random.choice(range(3, 14))
        data = [0] + [random.choice(range(3, 10)) for _ in range(random_num)] + [1]
        data += [2] * (max_length - len(data))
        src.append(data)
    src = torch.LongTensor(src)
    # decode的输入无需考虑最后一个
    tgt = src[:, :-1]

    # 作为label无需考虑第一个
    tgt_y = src[:, 1:]

    # 统计一共有多少个有效的值，便于后续计算平均loss
    n_token = torch.sum(src != 2)
    return src, tgt, tgt_y, n_token


def train(model, loss, optim, num=100, batch=32, device=torch.device("cpu")):
    model.train()

    def train_one():
        # 准备 data
        src, tgt, tgt_y, n_token = generate_random_batch(batch)
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_y = tgt_y.to(device)
        optim.zero_grad()
        out = model(src, tgt)
        out = model.predictor(out)
        # print(out.shape, tgt_y.shape)
        y_hat = out.contiguous().view(-1, out.size(-1))
        tgt_y = tgt_y.contiguous().view(-1)
        # print(y_hat.shape, tgt_y.shape)
        # TODO: 这种计算损失的方式，没有理解
        loss_value = loss(y_hat, tgt_y) / n_token
        loss_value.backward()
        optim.step()
        return loss_value.item()

    for i in range(num):
        loss_value = train_one()
        if i % 5 == 0:
            # print("loss value ", loss_value)
            print("Step {}, total_loss: {}".format(i, loss_value))


def predict(model, src):
    batch = src.size(0)
    tgt = torch.zeros(batch, 1).type(torch.long)
    model = model.eval()
    for i in range(15):
        out = model(src, tgt)
        out = model.predictor(out[:, -1])
        y_hat = out.argmax(dim=-1)
        tgt = torch.concat((tgt, y_hat.unsqueeze(1)), dim=-1)
    print(tgt)
    return tgt


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = CopyTaskModel(128)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)  # 3e-4
    # train(model, loss, optim, 2000, 32, device)
    # 保存模型的权重
    # torch.save(model.state_dict(), "model.pt")
    # # 加载模型的权重
    model.load_state_dict(torch.load("model.pt"))
    # src = torch.LongTensor([
    #     [0] + [3, 8, 9, 6, 7] * 10 + [1, 2]
    # ]).to(device)
    src, tgt, tgt_y, n_token = generate_random_batch(32)
    print(src)
    print('*' * 10 + "ans" + '*' * 10)
    tgt_hat = predict(model, src)

    print((src == tgt_hat).sum() / (src != -1).sum())
