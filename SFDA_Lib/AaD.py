import torch
import torch.nn as nn
from utils.tools import evaluation
from utils.model_utils import get_optimizer,get_scheduler
import torch.nn.functional as F
from utils.meters import AverageMeter
from SFDA_Lib.SFDA_Algorithm import SFDA_Algorithm

class AaD_Module(SFDA_Algorithm):
    def __init__(self, cfg, log, run, model_tar, model_src, tar_test_loader, src_test_loader):
        super(AaD_Module, self).__init__(cfg)
        self.cfg = cfg
        self.log = log 
        self.run = run
        self.model_tar = model_tar
        self.model_src = model_src
        self.tar_test_loader = tar_test_loader
        self.src_test_loader = src_test_loader
        self.Loss_FINAL = AverageMeter('loss_final', ':.4f')

    def _setup(self):
        self.acc_t0, self.mf1_t0, self.score_bank, self.fea_bank, self.labels, self.predicts0, \
            self.features_tar0 = evaluation(self.tar_test_loader, self.model_tar, self.device)
        cfg = self.cfg
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_tar, flag='tar')
        
        if cfg.wandb_log:
            wandb_data={'acc_t': self.acc_t0, 'mf1_t': self.mf1_t0}
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

    def forward(self, X_T, features_T, outputs_T, index_batch, iter_num, max_iters, epoch_idx, *args):
        cfg = self.cfg
        softmax_out = nn.Softmax(dim=1)(outputs_T)
        batch = X_T.size(0)
        if True:
            alpha = (1 + 10 * iter_num / max_iters) ** (-cfg.SFDA.beta) * cfg.SFDA.alpha
        else:
            alpha = cfg.SFDA.alpha
        with torch.no_grad():
            output_f_norm = F.normalize(features_T)
            output_f_ = output_f_norm.cpu().detach().clone().to(self.device)

            pred_bs = softmax_out

            self.fea_bank[index_batch] = output_f_.detach().clone().to(self.device)
            self.score_bank[index_batch] = softmax_out.detach().clone()

            distance = output_f_ @ self.fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=cfg.SFDA.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = self.score_bank[idx_near]  # batch x K x C

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, cfg.SFDA.K, -1
        )  # batch x K x C

        loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        ) # Equal to dot product

        mask = torch.ones((batch, batch))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T  # .detach().clone()#

        dot_neg = softmax_out @ copy  # batch x batch

        dot_neg = (dot_neg * mask.to(self.device)).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha

        self.Loss_FINAL.update(loss.item())
        return loss

    
    def meter_reset(self):
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