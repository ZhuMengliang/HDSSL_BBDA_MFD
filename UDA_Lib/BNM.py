#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import get_optimizer,get_scheduler
from UDA_Lib.UDA_Algorithm import UDA_Algorithm

class BNM_Module(UDA_Algorithm):
    def __init__(self, cfg, model_src, test_data_set, src_data_loader):
        super(BNM_Module, self).__init__(cfg)
        self.cfg = cfg
        self.model_src = model_src

    def _setup(self):
        cfg = self.cfg
        self.features_dim = cfg.Model_src.bottleneck_dim
        self.num_classes = cfg.Dataset.class_num
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_src, flag='src')
        self.bnm_loss = BatchNuclearnormMaximizationLoss().to(self.device)


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
        loss_reg = self.bnm_loss(outputs_T)
        loss_final = loss_cls + cfg.UDA.bnm*loss_reg 

        domain_acc = 0.0
        loss_tl = torch.tensor(0.0).to(device)
        return loss_final, loss_cls, loss_tl, loss_reg, domain_acc


class BatchNuclearnormMaximizationLoss(nn.Module):
    def __init__(self):
        super(BatchNuclearnormMaximizationLoss, self).__init__()

    def forward(self, output_t: torch.Tensor,) -> torch.Tensor:
        prob_t = nn.Softmax(dim=1)(output_t)
        _, s_tgt, _ = torch.svd(prob_t)
        loss = -torch.mean(s_tgt)       
        
        return loss