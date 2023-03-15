# dataset & dataloader

我们在 dataset.ipynb 内，实现了自定义的数据集，以及数据加载器。

one_hot 编码的方式：

```python
def ont_hot_encoding(x, num_classes):
    return torch.zeros(num_classes).scatter_(0, x, 1)

ont_hot_encoding(torch.tensor([1,3,5]), 10)
```
