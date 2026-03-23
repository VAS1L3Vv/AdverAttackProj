import torch

def fgsm_attack(model, x, eps, loss_fn):
    x_adv = x.clone().detach().requires_grad_(True)

    model.train()

    out = model(x_adv)
    loss = loss_fn(x, out)

    loss.backward()
    grad = x_adv.grad.sign()
    
    with torch.no_grad():
        x_adv = x_adv + eps * grad
        x_adv = torch.clamp(x_adv, 0, 1)

    model.eval()
    return x_adv.detach(), out

def pgd_attack(model, x, eps, loss_fn, steps=20):
    step_size = eps / 4
    x_adv = x.clone().detach()

    model.train()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        out = model(x_adv)
        loss = loss_fn(x, out)

        loss.backward()
        grad = x_adv.grad.sign()

        x_adv = x_adv + step_size * grad
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    model.eval()
    return x_adv.detach(), out

def apgd_attack(model, x, eps, loss_fn, steps=20, alpha=None, beta=0.75):
    if alpha is None:
        alpha = 2 * eps / steps

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)
    g = torch.zeros_like(x_adv)

    model.train()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        x_adv.grad = None

        out = model(x_adv)
        loss = loss_fn(x, out)

        loss.backward()
        grad_norm = x_adv.grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-8

        g = beta * g + x_adv.grad / grad_norm
        
        with torch.no_grad():
            x_adv += alpha * g.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)

    model.eval()
    return x_adv.detach(), out
