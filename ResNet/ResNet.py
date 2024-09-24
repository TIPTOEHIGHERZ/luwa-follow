import os
import torch
import torch.nn as nn
import torchvision


curr_dir = os.path.dirname(__file__)


def load_resnet101(num_class: int, device='cuda') -> nn.Module:
    resnet = torchvision.models.resnet101(weights=None)
    weights = torch.load(curr_dir + '/checkpoints/resnet101-63fe2227.pth')
    resnet.load_state_dict(weights)
    # 接头霸王
    resnet.fc = nn.Linear(2048, num_class)
    resnet.to(device=device)

    return resnet


if __name__ == '__main__':
    print(load_resnet101(10))
