import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import get_optimizer,get_scheduler, MultiModel
from UDA_Lib.UDA_Algorithm import UDA_Algorithm
from UDA_Lib.DANN import DomainAdversarialLoss
from models.Domain_Discriminator import DomainDiscriminator

class BSP_Module(UDA_Algorithm):
    def __init__(self, cfg, model_src, test_data_set, src_data_loader):
        super(BSP_Module, self).__init__(cfg)
        self.cfg = cfg
        self.model_src = model_src
        self.src_data_loader = src_data_loader

    def _setup(self):
        cfg = self.cfg
        self.features_dim = cfg.Model_src.bottleneck_dim
        self.num_classes = cfg.Dataset.class_num
        self.domain_discri = DomainDiscriminator(in_feature=self.features_dim, hidden_size=1024).to(self.device)
        multi_model = MultiModel([self.model_src, self.domain_discri])
        self.optimizer, self.scheduler = self._get_optimizer(model=multi_model, flag='src')

        from models.GRL import WarmStartGradientReverseLayer, GradientReverseLayer
        if cfg.UDA.adv_warmup:
            max_iters =  len(self.src_data_loader) # type: ignore
            max_iters =  len(self.src_data_loader)*cfg.Training.src_epoch
            grl = WarmStartGradientReverseLayer(max_iters=max_iters, auto_step=True)
        else:
            grl = GradientReverseLayer()

        self.domain_adv = DomainAdversarialLoss(self.domain_discri, grl=grl).to(self.device)
        self.bsp_penalty = BatchSpectralPenalizationLoss().to(self.device)

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
        loss_tl = self.domain_adv(features_S, features_T)
        loss_reg = self.bsp_penalty(features_S, features_T)
        loss_final = loss_cls + cfg.UDA.dann*loss_tl + cfg.UDA.bsp*loss_reg
        
        domain_acc = self.domain_adv.domain_discriminator_accuracy
        return loss_final, loss_cls, loss_tl, loss_reg, domain_acc



class BatchSpectralPenalizationLoss(nn.Module):
    r"""Batch spectral penalization loss from `Transferability vs. Discriminability: Batch
    Spectral Penalization for Adversarial Domain Adaptation (ICML 2019)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-penalization-icml19.pdf>`_.

    Given source features :math:`f_s` and target features :math:`f_t` in current mini batch, singular value
    decomposition is first performed

    .. math::
        f_s = U_s\Sigma_sV_s^T

    .. math::
        f_t = U_t\Sigma_tV_t^T

    Then batch spectral penalization loss is calculated as

    .. math::
        loss=\sum_{i=1}^k(\sigma_{s,i}^2+\sigma_{t,i}^2)

    where :math:`\sigma_{s,i},\sigma_{t,i}` refer to the :math:`i-th` largest singular value of source features
    and target features respectively. We empirically set :math:`k=1`.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    """

    def __init__(self):
        super(BatchSpectralPenalizationLoss, self).__init__()

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.svd(f_s)
        _, s_t, _ = torch.svd(f_t)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return loss
