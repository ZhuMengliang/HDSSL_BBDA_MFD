import torch
import torch.nn as nn
from utils.tools import evaluation
from utils.model_utils import get_optimizer, get_scheduler
import numpy as np
from utils.meters import AverageMeter
from SFDA_Lib.SFDA_Algorithm import SFDA_Algorithm
class SFCA_Module(SFDA_Algorithm):
    def __init__(self, cfg, log, run, model_tar, model_src, tar_test_loader, src_test_loader):
        super(SFCA_Module, self).__init__(cfg)
        self.cfg = cfg
        self.log = log 
        self.run = run
        self.model_tar = model_tar
        self.model_src = model_src
        self.tar_test_loader = tar_test_loader
        self.src_test_loader = src_test_loader

        self.LOSS_ST = AverageMeter('loss_st', ':.4f')
        self.Loss_IM = AverageMeter('loss_im', ':.4f')
        self.Loss_ICT = AverageMeter('loss_ict', ':.4f')
        self.Loss_FINAL = AverageMeter('loss_final', ':.4f')
    def _setup(self):
        self.acc_t0, self.mf1_t0, self.P_t0, self.features_tar_norm0, self.labels, self.predicts0, \
            self.features_tar0 = evaluation(self.tar_test_loader, self.model_tar, self.device)
        cfg = self.cfg
        # frozen the classifier module of the target model
        for k, v in self.model_tar.classifier.named_parameters():
            v.requires_grad = False
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_tar, flag='tar')
        if cfg.wandb_log:
            wandb_data={'acc_t': self.acc_t0,'mf1_t': self.mf1_t0}
            self._wandb_log(data=wandb_data, step=0)
        if cfg.logging:
            self.log.info(f'Seed: {cfg.seed}, '
                        f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                        f'Initialized Target Model, '
                        f'Acc_t: {self.acc_t0:.4f}, ')
    def _wandb_log(self,data,step):
            self.run.log(data=data, step=step)
    def _get_optimizer(self, model, flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Opt_src if flag=='src' else cfg.Opt_tar        
        optimizer=get_optimizer(args, model)
        scheduler=get_scheduler(args, optimizer)
        return optimizer, scheduler
    def _IM_loss(self, outputs):
        probs = nn.Softmax(dim=1)(outputs)
        entropy = -probs * torch.log(probs + 1e-5)
        entropy = torch.sum(entropy, dim=1)
        entropy = torch.mean(entropy)
        probs_ = probs.mean(dim=0)
        diversity = -torch.sum(-probs_ * torch.log(probs_ + 1e-5))
        loss_IM = entropy + diversity
        return loss_IM
    def update_batch_stats(self, model, flag):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.update_batch_stats = flag
    def forward(self, X_T, features_T, outputs_T, index_batch, iter_num, max_iters, epoch_idx, *args):
        cfg = self.cfg
        batch = X_T.size(0)
        probs_T = nn.Softmax(dim=1)(outputs_T)
        probs_T_max, pseudo_labels = torch.max(probs_T, dim=1)
        index_certain = probs_T_max >= cfg.SFDA.threshold
        batch_certain = int(index_certain.float().sum().cpu().numpy())
        if batch_certain > 0:
            pseudo_labels = pseudo_labels[index_certain].detach().to(self.device)
            loss_st = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs_T[index_certain], pseudo_labels)
        else:
            loss_st = torch.tensor(0.0).to(self.device)

        loss_im = self._IM_loss(outputs_T)

        lam = np.random.beta(cfg.SFDA.alpha, cfg.SFDA.alpha)
        index = torch.randperm(batch).to(self.device)
        mixed_input = lam * X_T + (1 - lam) * X_T[index, :]
        mixed_probs = (lam * probs_T + (1 - lam) * probs_T[index, :]).detach()
        self.update_batch_stats(self.model_tar, False)
        outputs_T_mixed = self.model_tar(mixed_input)[0]
        self.update_batch_stats(self.model_tar, True)
        probs_T_mixed = nn.Softmax(dim=1)(outputs_T_mixed)
        loss_ict = nn.MSELoss()(probs_T_mixed, mixed_probs)

        loss_final = loss_st + loss_im + loss_ict
        self.LOSS_ST.update(loss_st.item(), n=batch_certain)
        self.Loss_IM.update(loss_im.item(), n=batch)
        self.Loss_ICT.update(loss_ict.item(), n=batch)
        self.Loss_FINAL.update(loss_final.item(), n=batch)
        return loss_final
    def meter_reset(self):
        self.LOSS_ST.reset()
        self.Loss_IM.reset()
        self.Loss_ICT.reset()
        self.Loss_FINAL.reset()
    def epoch_evaluation(self, epoch_idx, train_memory, train_time):
        cfg = self.cfg
        self.model_tar.eval()
        self.acc_t, self.mf1_t = evaluation(self.tar_test_loader, self.model_tar, self.device)[0:2]
        self.acc_t_src, self.mf1_t_src = evaluation(self.src_test_loader, self.model_tar, self.device)[0:2]
        self.lr = self.optimizer.param_groups[0]['lr']
        if cfg.wandb_log:
            wandb_data={
                        'acc_t': self.acc_t, 
                        'mf1_t': self.mf1_t, 
                        'mf1_t_src': self.mf1_t_src, 
                        'acc_t_src': self.acc_t_src,
                        'loss_st': self.LOSS_ST.avg,
                        'loss_im': self.Loss_IM.avg,
                        'loss_ict': self.Loss_ICT.avg,
                        'loss_final': self.Loss_FINAL.avg,
                        'train_memory': train_memory,
                        'train_time': train_time,
                        'lr': self.lr
                        }
            self._wandb_log(data=wandb_data, step=epoch_idx+1)
        if cfg.logging:
            self.log.info(
                        f'Seed:{cfg.seed},'
                        f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                        f'Epoch:{epoch_idx+1}/{cfg.Training.tar_epoch},'
                        f'Acc_t:{self.acc_t:.4f},'
                        f'Acc_s:{self.acc_t_src:.4f},'
                        f'Train_Memory:{train_memory:.4f}MB,'
                        f'Train_Time:{train_time:.4f}s'
                        )
        return 