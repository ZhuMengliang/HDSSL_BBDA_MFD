from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GRL import WarmStartGradientReverseLayer, GradientReverseLayer
from utils.metric import binary_accuracy, accuracy
from utils.model_utils import get_optimizer,get_scheduler, MultiModel
from UDA_Lib.UDA_Algorithm import UDA_Algorithm
from models.Domain_Discriminator import DomainDiscriminator
class DANN_Module(UDA_Algorithm):
    def __init__(self, cfg, model_src, test_data_set, src_data_loader):
        super(DANN_Module, self).__init__(cfg)
        self.cfg = cfg
        self.model_src = model_src
        self.src_data_loader = src_data_loader
        self.test_data_set = test_data_set
    def _setup(self):
        cfg = self.cfg
        self.features_dim = cfg.Model_src.bottleneck_dim
        self.num_classes = cfg.Dataset.class_num
        self.domain_discri = DomainDiscriminator(in_feature=self.features_dim, hidden_size=1024).to(self.device)
        multi_model = MultiModel([self.model_src, self.domain_discri])
        self.optimizer, self.scheduler = self._get_optimizer(model=multi_model, flag='src')
        if cfg.UDA.adv_warmup:
            max_iters =  len(self.src_data_loader) 
            max_iters =  len(self.src_data_loader)*cfg.Training.src_epoch
            grl = WarmStartGradientReverseLayer(max_iters=max_iters, auto_step=True)
        else:
            grl = GradientReverseLayer()
        self.domain_adv = DomainAdversarialLoss(self.domain_discri, grl=grl).to(self.device)
    def _get_optimizer(self, model, flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Opt_src if flag=='src' else cfg.Opt_tar        
        optimizer=get_optimizer(args, model)
        scheduler=get_scheduler(args, optimizer)
        return optimizer, scheduler
    def forward(self, outputs, features, Y_S, *args):
        cfg = self.cfg
        device = self.device
        outputs_S, outputs_T = outputs.chunk(2, dim=0)
        features_S, features_T = features.chunk(2, dim=0)
        loss_cls = F.cross_entropy(outputs_S, Y_S)
        loss_tl = self.domain_adv(features_S, features_T)
        loss_final = loss_cls + cfg.UDA.dann*loss_tl 
        loss_reg = torch.tensor(0.0).to(device)
        domain_acc = self.domain_adv.domain_discriminator_accuracy
        return loss_final, loss_cls, loss_tl, loss_reg, domain_acc
class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                grl = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction) # type: ignore
        self.domain_discriminator_accuracy = None
    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) + # type: ignore
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction) # type: ignore
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)

