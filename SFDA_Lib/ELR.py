import torch
import torch.nn as nn
from utils.tools import evaluation
from utils.model_utils import get_optimizer,get_scheduler
import torch.nn.functional as F
from utils.meters import AverageMeter
from SFDA_Lib.SFDA_Algorithm import SFDA_Algorithm

class ELR_Module(SFDA_Algorithm):
    def __init__(self, cfg, log, run, model_tar, model_src, tar_test_loader, src_test_loader):
        super(ELR_Module, self).__init__(cfg)
        self.cfg = cfg
        self.log = log 
        self.run = run
        self.model_tar = model_tar
        self.model_src = model_src
        self.tar_test_loader = tar_test_loader
        self.src_test_loader = src_test_loader
        self.Loss_ELR = AverageMeter('loss_elr', ':.4f')
        self.Loss_NRC = AverageMeter('loss_nrc', ':.4f')
        self.Loss_FINAL = AverageMeter('loss_final', ':.4f')

    def _setup(self):
        self.acc_t0, self.mf1_t0, self.score_bank, self.fea_bank, self.labels, self.predicts0, \
            self.features_tar0 = evaluation(self.tar_test_loader, self.model_tar, self.device)
        cfg = self.cfg
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_tar, flag='tar')

        self.elr_loss = ELR_loss(beta=cfg.SFDA.beta, lamb=cfg.SFDA.lamb, num=self.score_bank.shape[0], 
                                cls=cfg.Dataset.class_num, device=self.device)
        
        if cfg.wandb_log:
            wandb_data={'acc_t': self.acc_t0, 'mf1_t': self.mf1_t0, }
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
        with torch.no_grad():
            output_f_norm = F.normalize(features_T)
            output_f_ = output_f_norm.cpu().detach().clone().to(self.device)

            self.fea_bank[index_batch] = output_f_.detach().clone().to(self.device)
            self.score_bank[index_batch] = softmax_out.detach().clone()

            distance = output_f_@self.fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=cfg.SFDA.K+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = self.score_bank[idx_near]    #batch x K x C

            fea_near = self.fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = self.fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
            _, idx_near_near = torch.topk(distance_,dim=-1,largest=True,k=cfg.SFDA.KK+1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:] # batch x K x M
            index_batch_ = index_batch.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == index_batch_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1, cfg.SFDA.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = self.score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                            outputs_T.size(1))  # batch x KM x C
            score_self = self.score_bank[index_batch]

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, cfg.SFDA.K * cfg.SFDA.KK, -1)  # batch x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
            weight_kk.to(self.device)).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        loss_nrc = torch.mean(const)
        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, cfg.SFDA.K,-1)  # batch x K x C

        loss_nrc += torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
            weight.to(self.device)).sum(1))

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        #loss += -torch.mean((softmax_out * score_self).sum(-1))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *torch.log(msoftmax + 1e-8))

        loss_nrc += gentropy_loss
        loss_elr = self.elr_loss(index_batch, outputs_T)
        loss_final = loss_nrc + loss_elr

        self.Loss_NRC.update(loss_nrc.item())
        self.Loss_ELR.update(loss_elr.item())
        self.Loss_FINAL.update(loss_final.item())
        return loss_final

    
    def meter_reset(self):
        self.Loss_NRC.reset()
        self.Loss_ELR.reset()
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
                        'loss_nrc': self.Loss_NRC.avg,
                        'loss_elr': self.Loss_ELR.avg,
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
    

class ELR_loss(nn.Module):
    def __init__(self, beta, lamb, num, cls, device):
        super(ELR_loss, self).__init__()
        # self.pseudo_targets = torch.empty(args.nb_samples, dtype=torch.long).random_(args.nb_classes).to(self.device)
        self.device = device
        self.ema = torch.zeros(num, cls).to(self.device)
        self.beta = beta
        self.lamb = lamb


    def forward(self, index,  outputs):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()

        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg
        return final_loss