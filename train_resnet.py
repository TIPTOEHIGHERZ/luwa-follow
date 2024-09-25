import torch
import torch.nn as nn
import tqdm
from data import BatchLoader
from typing import Callable
from ResNet import load_resnet101
from torch.utils.data import DataLoader


def calc_acc(y_pred: torch.Tensor, y: torch.Tensor):
    y_pred = y_pred.detach()
    y_pred = torch.argmax(y_pred, dim=-1)
    return torch.sum(torch.eq(y_pred, y).int()) / y.shape[0]


def train(model: nn.Module,
          data_loader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          loss_fn: Callable = nn.CrossEntropyLoss(),
          lr_scheduler=None,
          device='cuda'):
    data = tqdm.tqdm(data_loader)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for x, y in data:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()

            acc = calc_acc(y_pred, y)
            data.set_description(f'{epoch} / {epochs}: ')
            data.set_postfix({'acc': acc.item(), 'loss': loss.item()})

        if lr_scheduler:
            lr_scheduler.step()

    return


device = 'cuda'
lr = 1e-4
epochs = 1
batch_size = 32
batch_loader = BatchLoader()
# means, stds = batch_loader.get_statistics()
# print(means, stds)
means, stds = torch.tensor([0.2249, 0.2249, 0.2249]), torch.tensor([0.1403, 0.1403, 0.1403])
batch_loader.prepare_transform(means, stds)
data_loader = DataLoader(batch_loader, batch_size=batch_size, num_workers=4, shuffle=True)
print(f'total class num: {batch_loader.class_num}')
resnet = load_resnet101(batch_loader.class_num)
resnet.to(device)
optimizer = torch.optim.Adam(params=[
    {'params': resnet.fc.parameters(), 'lr': lr * 10}
], lr=lr)

train(resnet, batch_loader, optimizer, epochs, device=device)

