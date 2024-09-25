import torch


def calc_acc(y_pred: torch.Tensor, y: torch.Tensor):
    y_pred = y_pred.detach()
    y_pred = torch.argmax(y_pred, dim=-1)
    return torch.sum(torch.eq(y_pred, y).int()) / y.shape[0]
