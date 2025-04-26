#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import get_optimizer,get_scheduler
from UDA_Lib.UDA_Algorithm import UDA_Algorithm

class SAFN_Module(UDA_Algorithm):
    def __init__(self, cfg, model_src, test_data_set, src_data_loader):
        super(SAFN_Module, self).__init__(cfg)
        self.cfg = cfg
        self.model_src = model_src

    def _setup(self):
        cfg = self.cfg
        self.features_dim = cfg.Model_src.bottleneck_dim
        self.num_classes = cfg.Dataset.class_num
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_src, flag='src')
        self.safn_loss = StepwiseAdaptiveFeatureNormLoss().to(self.device)


    def _get_optimizer(self, model, flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Opt_src if flag=='src' else cfg.Opt_tar        
        optimizer=get_optimizer(args, model)
        scheduler=get_scheduler(args, optimizer)
        #还少个scheduler
        return optimizer, scheduler

    def forward(self, outputs, features, Y_S, *args):
        cfg = self.cfg
        device = self.device
        outputs_S, outputs_T = outputs.chunk(2, dim=0)
        features_S, features_T = features.chunk(2, dim=0)
        loss_cls = F.cross_entropy(outputs_S, Y_S)
        loss_reg = self.safn_loss(features_S, features_T)
        loss_final = loss_cls + cfg.UDA.safn*loss_reg 

        domain_acc = 0.0
        loss_tl = torch.tensor(0.0).to(device)
        return loss_final, loss_cls, loss_tl, loss_reg, domain_acc


'''
Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptatio
'''
class StepwiseAdaptiveFeatureNormLoss(nn.Module):
    def __init__(self):
        super(StepwiseAdaptiveFeatureNormLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        loss = self.get_L2norm_loss(f_s) + self.get_L2norm_loss(f_t)
        return loss

    def get_L2norm_loss(self, x):
        radius = x.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + 1.0
        l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
        return l