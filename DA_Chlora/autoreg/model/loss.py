import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def mae_loss(output, target):
    return F.l1_loss(output, target)

def mse_loss_with_mask(output, target, mask):
    return F.mse_loss(output * mask, target * mask)

def mae_loss_with_mask(output, target, mask):
    return F.l1_loss(output * mask, target * mask)
