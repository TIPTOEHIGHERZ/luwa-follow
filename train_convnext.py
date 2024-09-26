#! /bin/python3
import torch
import torch.nn as nn
import tqdm
from data import BatchLoader
from typing import Callable
from ConvNext import load_convnext_base
from torch.utils.data import DataLoader
from utils import calc_acc
import pathlib, os


torch.manual_seed(1234)


def train(model: nn.Module,
          data_loader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          loss_fn: Callable = nn.CrossEntropyLoss(),
          lr_scheduler=None,
          device='cuda',
          save_period=5,
          save_path=None):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_acc = 0.0
        total_loss = 0.0
        total_data = 0
        data = tqdm.tqdm(data_loader)

        for x, y in data:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            total_data += x.shape[0]

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()

            acc = calc_acc(y_pred, y)
            total_acc += acc.item() * x.shape[0]
            total_loss += loss.item() * x.shape[0]

            data.set_description(f'EPOCHES: {epoch} / {epochs}: ')
            data.set_postfix({'acc': total_acc / total_data, 'loss': total_loss / total_data})

        if lr_scheduler:
            lr_scheduler.step()

        if (epoch + 1) % save_period == 0:
            torch.save(model.state_dict(), save_path.joinpath(f'ckpt_{epoch + 1}.pth'))

    return


if __name__ == '__main__':
    device = 'cuda'
    lr = 1e-4
    epochs = 50
    batch_size = 32

    meta_data = ['256', 'train', '20x', '6w', 'texture']

    batch_loader = BatchLoader(*meta_data)
    # means, stds = batch_loader.get_statistics()
    # print(means, stds)
    means, stds = torch.tensor([0.2249, 0.2249, 0.2249]), torch.tensor([0.1403, 0.1403, 0.1403])
    batch_loader.prepare_transform(means, stds)
    data_loader = DataLoader(batch_loader, batch_size=batch_size, num_workers=4, shuffle=True)
    print(f'total class num: {batch_loader.class_num}')
    convnext = load_convnext_base(batch_loader.class_num)
    # resnet.load_state_dict(torch.load('checkpoints/ckpt_49.pth', weights_only=False))
    convnext.to(device)
    optimizer = torch.optim.Adam(params=[
        {'params': convnext.classifier[-1].parameters(), 'lr': lr * 10}
    ], lr=lr)

    p = pathlib.Path(os.path.join(os.path.dirname(__file__), 'checkpoints'))
    p.mkdir(exist_ok=True)
    p = p.joinpath('_'.join(meta_data))
    p.mkdir(exist_ok=True)

    train(convnext, data_loader, optimizer, epochs, device=device, save_path=p, save_period=2)

