import torch
import torch.nn as nn
import tqdm
from data import BatchLoader
from typing import Callable
from ResNet import load_resnet101
from ConvNext import load_convnext_base
from torch.utils.data import DataLoader
from utils import calc_acc
import os, pathlib


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
    import argparse

    model_type = ['ConvNext', 'ResNet']

    parser = argparse.ArgumentParser('train')
    parser.add_argument('model_type',
                        help=f'model type in {model_type}', type=str)

    args = parser.parse_args()
    if args.model_type not in model_type:
        print('model not support')
        exit(0)

    device = 'cuda'
    batch_size = 32

    meta_data = ['256', 'test', '20x', '6w', 'texture']

    batch_loader = BatchLoader(*meta_data)
    print(f'class num: {batch_loader.class_num}')
    means, stds = torch.tensor([0.2249, 0.2249, 0.2249]), torch.tensor([0.1403, 0.1403, 0.1403])
    batch_loader.prepare_transform(means, stds)
    data_loader = DataLoader(batch_loader, shuffle=True, batch_size=batch_size)

    if args.model_type == 'ConvNext':
        model = load_convnext_base(batch_loader.class_num)
        last_layer = model.classifier[-1]
    elif args.model_type == 'ResNet':
        model = load_resnet101(batch_loader.class_num)
        last_layer = model.fc

    model.to(device)

    meta_data[1] = 'train'
    p = pathlib.Path(os.path.join(os.path.dirname(__file__), args.model_type, 'checkpoints', '_'.join(meta_data)))
    ckpt_list = os.listdir(p)

    for ckpt in ckpt_list[0:]:
        if len(ckpt) > 4 and ckpt[-4:] == '.pth':
            model.load_state_dict(torch.load(p.joinpath(ckpt), weights_only=False))
            # for param in resnet.fc.parameters():
            #     print(param)
            evaluate(model, data_loader)
