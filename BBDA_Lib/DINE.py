import torch.nn.functional as F
import torch
import torch.nn as nn

def IM_loss(probs):
    entropy = -probs * torch.log(probs + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    entropy = torch.mean(entropy)
    probs_ = probs.mean(dim=0)
    diversity = -torch.sum(-probs_ * torch.log(probs_ + 1e-5))
    loss_IM = entropy + diversity
    return loss_IM


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.update_batch_stats = flag