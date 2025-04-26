#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import get_optimizer,get_scheduler
from UDA_Lib.UDA_Algorithm import UDA_Algorithm

class ATDOC_NA_Module(UDA_Algorithm):
    def __init__(self, cfg, model_src, test_data_set, src_data_loader):
        super(ATDOC_NA_Module, self).__init__(cfg)
        self.cfg = cfg
        self.model_src = model_src
        self.test_data_set = test_data_set

    def _setup(self):
        cfg = self.cfg
        self.features_dim = cfg.Model_src.bottleneck_dim
        self.num_classes = cfg.Dataset.class_num
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_src, flag='src')
        tar_data_size = len(self.test_data_set)
        mem_features = torch.rand(tar_data_size, self.features_dim).to(self.device)
        self.mem_features = mem_features / torch.norm(mem_features, p=2, dim=1, keepdim=True)
        self.mem_class = torch.ones(tar_data_size, self.num_classes).to(self.device) / self.num_classes

    def _get_optimizer(self, model, flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Opt_src if flag=='src' else cfg.Opt_tar        
        optimizer=get_optimizer(args, model)
        scheduler=get_scheduler(args, optimizer)
        #还少个scheduler
        return optimizer, scheduler

    def forward(self, outputs, features, Y_S, X_S, X_T, index_S, index_T):
        cfg = self.cfg
        device = self.device
        outputs_S, outputs_T = outputs.chunk(2, dim=0)
        features_S, features_T = features.chunk(2, dim=0)
        loss_cls = F.cross_entropy(outputs_S, Y_S)

        dis = -torch.mm(features_T.detach(), self.mem_features.t()) # B*N
        for di in range(dis.size(0)):
            dis[di, index_T[di]] = torch.max(dis) #把自己的相似度设为最大，防止自己和自己匹配
        _, pl = torch.sort(dis, dim=1) # B*N 从小到大排序，

        w = torch.zeros(features_T.size(0), self.mem_features.size(0)).to(self.device)
        for wi in range(w.size(0)):
            for wj in range(cfg.UDA.K):
                w[wi][pl[wi, wj]] = 1/ cfg.UDA.K
        weight_, pred = torch.max(w.mm(self.mem_class), 1)
        loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_T, pred)
        loss_tl = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())  
        loss_final = loss_cls + cfg.UDA.atdoc_na*loss_tl
        
        # update memory的方式跟源代码中的有所出入，
        # 源代码是在完成模型当前iteration更新后，再重新将样本输入到模型中得到对应的特征和概率预测，再更新memory bank
        # 实际上大部分UDA或SFDA 以batch-wise方式更新memory bank时，都是直接更新
        with torch.no_grad():
            features_T = features_T.detach() / torch.norm(features_T, p=2, dim=1, keepdim=True)
            softmax_out = nn.Softmax(dim=1)(outputs_T)
            outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))
        self.mem_features[index_T] =  features_T.detach().clone()
        self.mem_class[index_T] =  outputs_target.detach().clone()

        domain_acc = 0.0
        loss_reg = torch.tensor(0.0).to(device)
        return loss_final, loss_cls, loss_tl, loss_reg, domain_acc