import torch
import torch.nn.functional as F
from autoattack import AutoAttack

device = "cuda" if torch.cuda.is_available() else "cpu"

def fgsm_attack(model, x, eps, loss_fn):
    model.train()
    x_adv = x.clone().detach().requires_grad_(True)

    out = model(x_adv)
    loss = loss_fn(x, out)
    loss.backward()

    grad = x_adv.grad.sign()
    x_adv = x_adv + eps * grad
    x_adv = torch.clamp(x_adv, 0, 1)

    model.eval()

    return x_adv.detach(), out

def pgd_attack(model, x, eps, loss_fn, steps=50):
    model.train()

    step_size = eps / 4
    x_adv = x.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = model(x_adv)
        loss = loss_fn(x, out)

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.sign()

        x_adv = x_adv + step_size * grad
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    model.eval()
    return x_adv, out

def apgd_attack(model, x, eps, loss_fn, steps=40, alpha=None):
    if alpha is None:
        alpha = 2 * eps / steps

    x_adv = x.clone().detach().to(x.device)
    x_adv.requires_grad_(True)

    model.train()

    g = torch.zeros_like(x_adv)
    beta = 0.75

    for _ in range(steps):
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        out = model(x_adv)
        loss = loss_fn(x, out)
        loss.backward()

        grad_norm = x_adv.grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-8
        g = beta * g + x_adv.grad / grad_norm
        x_adv.data = x_adv.data + alpha * g.sign()

        x_adv.data = torch.min(torch.max(x_adv.data, x - eps), x + eps)
        x_adv.data = torch.clamp(x_adv.data, 0, 1)

    return x_adv.detach(), out
