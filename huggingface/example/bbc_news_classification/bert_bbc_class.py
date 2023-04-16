# BBC 文章分类
# 数据集来源: https://www.kaggle.com/competitions/learn-ai-bbc/
# 代码参考: https://mp.weixin.qq.com/s/00on_zUFjAmnoSb_8j0QMw

# 下一步: https://xiuweihan.cn/2021/04/15/2021-04-15-nlp%E4%B9%8Btransformers%E5%BA%93/
import torch
from torch import nn
from transformers import BertModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'batch': 1,
    'epochs': 1,
    'lr': 3e-4,
    'device': device,
    'bert_path': 'bert-base-cased',
    'train_path': 'data/BBC News Train.csv',
    # 'train_path': 'data/demo.csv',
    'test_path': 'data/BBC News Test.csv',
    'model_path': 'model/bert_bbc_class.pth'
}


def demo():
    BERT_PATH = 'bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    bert = BertModel.from_pretrained(BERT_PATH)

    example_text = 'I am watching Memento tonight.'
    tokens = tokenizer(example_text,
                       padding='max_length',
                       max_length=10,
                       truncation=True,
                       return_tensors="pt")


labels = {
    'business': 0,
    'entertainment': 1,
    'sport': 2,
    'tech': 3,
    'politics': 4
}

label_name = {v: k for k, v in labels.items()}


class SampleDataset(Dataset):
    def __init__(self, df,
                 is_train=True,
                 tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')):
        self.is_train = is_train
        if is_train:
            self.labels = [labels[label] for label in df['Category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['Text']]

    def classes(self):
        assert self.is_train
        return self.labels

    def __len__(self):
        return len(self.texts)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        assert self.is_train
        # return torch.tensor([self.labels[idx]])
        return self.labels[idx]

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        if not self.is_train:
            return self.get_batch_texts(idx)
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    # def forward(self, input_id, mask):
    def forward(self, input):
        output = self.bert(**input)
        pooler_output = output.pooler_output
        dropout_output = self.dropout(pooler_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def get_dataset(ratio=[0.8, 0.1, 0.1]):
    df = pd.read_csv(config['train_path'])
    dataset = SampleDataset(df)
    # torch切分数据集
    trainset, validset, testset = random_split(dataset, ratio)
    return trainset, validset, testset


def run_one_epoch(model, dataloader, is_train, criterion, optimizer, device, batch=config['batch']):
    def run():
        start = time.time()
        total_loss = 0
        total_acc = 0
        nums = len(dataloader) * batch
        for feature, label in tqdm(dataloader):
            label = label.to(device)
            if is_train:
                optimizer.zero_grad()
            input = {k: v.squeeze(1).to(device) for k, v in feature.items()}
            # TODO: 为何经过dataloader之后，中间会多一个维度
            output = model(input)
            loss = criterion(output, label)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            acc = (output.argmax(dim=-1) == label.view(-1)).sum().item()
            total_acc += acc
        end = time.time()
        print(end - start)
        return total_loss / nums, total_acc / nums

    if is_train:
        model.train()
        return run()
    if not is_train:
        model.eval()
        with torch.no_grad():
            return run()


def train(model, train_dataset, val_dataset,
          learning_rate=config['lr'],
          batch=config['batch'],
          epochs=config['epochs'],
          device=device):
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch_num in range(epochs):
        train_loss, train_acc = run_one_epoch(model, train_dataloader, True, criterion, optimizer, device, batch)
        val_loss, val_acc = run_one_epoch(model, val_dataloader, False, criterion, optimizer, device, batch)
        print(f'Epoch: {epoch_num + 1}')
        print(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')
        print(f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')


def predict(model, device=device):
    # 加载模型
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    test_dataset = SampleDataset(pd.read_csv(config['test_path']), is_train=False)
    # 加载测试集
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    tensors = []
    model.eval()
    with torch.no_grad():
        for features in tqdm(test_dataloader):
            mask = features['attention_mask'].to(device)
            input_id = features['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            y_pred = output.argmax(dim=1)
            tensors.append(y_pred.view(-1))
    t = torch.concat(tensors, dim=-1).tolist()
    ans = [label_name[key] for key in t]
    return ans


if __name__ == '__main__':
    # model = BertClassifier().to(device=device)
    # trainset, validset, testset = get_dataset([0.01, 0.01, 0.98])
    # train(model, trainset, validset, epochs=2, device=device, batch=2)


    def demo1():
        model = BertClassifier().to(device=device)
        trainset, validset, testset = get_dataset()

        train_dataloader = DataLoader(trainset, batch_size=4, shuffle=True)
        for feature, label in train_dataloader:
            input = {k: v.squeeze(1).to(device) for k, v in feature.items()}
            for v in input.values():
                print(v.shape)
            res = model(input)
            label = label.to(device)
            print(res.shape, label.shape)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(res, label)
            print(loss)

            break


    demo1()

    # train(model, trainset, validset, epochs=1)
    # torch.save(model.state_dict(), config['model_path'])
    # # evaluate(model, testset)
    # ans = predict(model)
    # print(ans)

    def demo():
        model = BertClassifier().to(device)
        sentence = "I love chrome."
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        input = tokenizer(sentence,
                          padding='max_length',
                          max_length=512,
                          truncation=True,
                          return_tensors="pt")
        # 将input转到gpu上
        input = {k: v.to(device) for k, v in input.items()}
        for v in input.values():
            print(v.shape)
        res = model(input)
        label = torch.tensor([1]).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(res, label)
        print(loss)

    # demo()
