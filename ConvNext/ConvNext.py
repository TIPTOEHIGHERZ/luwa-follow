import torch
import torchvision
import os
import torch.nn as nn


file_path = os.path.dirname(__file__)


def load_convnext_base(num_class: int, pretrained=True, device='cuda'):
    model = torchvision.models.convnext_base(weights=None)

    if pretrained:
        model.load_state_dict(torch.load(
            os.path.join(file_path, 'checkpoints', 'convnext_base-6075fbad.pth'), weights_only=False))

    last_in_features = model.classifier[-1].weight.shape[1]
    model.classifier[-1] = nn.Linear(last_in_features, num_class)
    model.to(device)
    return model


if __name__ == '__main__':
    model = load_convnext_base(6)
    print(model)
