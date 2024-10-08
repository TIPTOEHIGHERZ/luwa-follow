import os
import torch
import torch.nn as nn
import torchvision


curr_dir = os.path.dirname(__file__)


def load_resnet101(num_class: int, pretrained=True, device='cuda'):
    resnet = torchvision.models.resnet101(weights=None)
    if pretrained:
        weights = torch.load(curr_dir + '/checkpoints/resnet101-63fe2227.pth', weights_only=False)
        resnet.load_state_dict(weights)
    # 接头霸王
    resnet.fc = nn.Linear(2048, num_class)
    resnet.to(device=device)

    return resnet


if __name__ == '__main__':
    print(load_resnet101(10))
