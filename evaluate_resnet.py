import torch
import torch.nn as nn
import tqdm
from data import BatchLoader
from typing import Callable
from ResNet import load_resnet101
from torch.utils.data import DataLoader
from utils import calc_acc
import os


torch.manual_seed(1234)


def evaluate(model: nn.Module,
             data_loader,
             loss_fn: Callable = nn.CrossEntropyLoss(),
             device='cuda'):
    model.to(device)
    model.eval()
    data = tqdm.tqdm(data_loader)
    total_acc = 0.0
    total_loss = 0.0
    total_data = 0

    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            total_data += x.shape[0]

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            acc = calc_acc(y_pred, y)
            total_acc += acc.item() * x.shape[0]
            total_loss += loss.item() * x.shape[0]

            data.set_postfix({'acc': total_acc / total_data, 'loss': total_loss / total_data})

    return


if __name__ == '__main__':
    device = 'cuda'
    lr = 1e-4
    epochs = 50
    batch_size = 32
    batch_loader = BatchLoader(train='test')
    data_loader = DataLoader(batch_loader, shuffle=True)
    resnet = load_resnet101(batch_loader.class_num, pretrained=False)

    ckpt_list = os.listdir('./checkpoints')
    for ckpt in ckpt_list:
        if len(ckpt) > 4 and ckpt[-4:] == '.pth':
            resnet.load_state_dict(torch.load(f'./checkpoints/{ckpt}'))

    evaluate(resnet, data_loader)
