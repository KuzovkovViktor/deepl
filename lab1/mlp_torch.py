"""Еталонна реалізація в PyTorch (лише для звірки)."""
import numpy as np
import torch
import torch.nn.functional as F


def torch_loss_and_grads(X, y, p):
    torch.set_default_dtype(torch.float64)
    l1 = torch.nn.Linear(4, 8)
    l2 = torch.nn.Linear(8, 3)
    with torch.no_grad():
        # nn.Linear зберігає ваги як (out, in) -> транспонуємо
        l1.weight.copy_(torch.from_numpy(p["W1"].T.copy()))
        l1.bias.copy_(torch.from_numpy(p["b1"].copy()))
        l2.weight.copy_(torch.from_numpy(p["W2"].T.copy()))
        l2.bias.copy_(torch.from_numpy(p["b2"].copy()))
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).long()
    loss = F.cross_entropy(l2(torch.relu(l1(Xt))), yt)   # mean за замовчуванням
    loss.backward()
    grads = {
        "W1": l1.weight.grad.numpy().T.copy(),
        "b1": l1.bias.grad.numpy().copy(),
        "W2": l2.weight.grad.numpy().T.copy(),
        "b2": l2.bias.grad.numpy().copy(),
    }
    return loss.item(), grads
