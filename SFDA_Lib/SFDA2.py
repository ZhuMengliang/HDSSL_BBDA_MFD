import torch
import torch.nn as nn
from utils.tools import evaluation
from utils.model_utils import get_optimizer,get_scheduler
import torch.nn.functional as F
from utils.meters import AverageMeter
from SFDA_Lib.SFDA_Algorithm import SFDA_Algorithm

class SFDA2_Module(SFDA_Algorithm):
    def __init__(self, cfg, log, run, model_tar, model_src, tar_test_loader, src_test_loader):
        super(SFDA2_Module, self).__init__(cfg)
        self.cfg = cfg
        self.log = log 
        self.run = run
        self.model_tar = model_tar
        self.model_src = model_src
        self.tar_test_loader = tar_test_loader
        self.src_test_loader = src_test_loader
        self.Loss_POS = AverageMeter('loss_ps', ':.4f')
        self.Loss_NEG = AverageMeter('loss_neg', ':.4f')
        self.Loss_IFA = AverageMeter('loss_ifa', ':.4f')
        self.Loss_FD = AverageMeter('loss_fd', ':.4f')
        self.Loss_FINAL = AverageMeter('loss_final', ':.4f')

    def _setup(self):
        self.acc_t0, self.mf1_t0, self.score_bank, self.fea_bank, self.labels, self.pseudo_bank, \
            self.features_tar0 = evaluation(self.tar_test_loader, self.model_tar, self.device)
        cfg = self.cfg
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_tar, flag='tar')

        self.rho = torch.ones([cfg.Dataset.class_num]).to(self.device) / cfg.Dataset.class_num
        self.Cov = torch.zeros(cfg.Dataset.class_num, self.fea_bank.size(1), self.fea_bank.size(1)).to(self.device)
        self.Ave = torch.zeros(cfg.Dataset.class_num, self.fea_bank.size(1)).to(self.device)
        self.Amount = torch.zeros(cfg.Dataset.class_num).to(self.device)

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
        score_test = nn.Softmax(dim=1)(outputs_T)
        batch = X_T.size(0)
        alpha = (1 + 10 * iter_num / max_iters) ** (-cfg.SFDA.beta) * cfg.SFDA.alpha
        #源码采用的是WN分类层
        # w = F.normalize(self.model_tar.classifier.weight)
        # w = F.normalize(self.model_tar.classifier.weight)
        output_f_norm = F.normalize(features_T)
        w = self.model_tar.classifier.weight
        pseudo_label = torch.argmax(score_test, 1).detach()
        top2 = torch.topk(score_test, 2).values
        margin = top2[:,0] - top2[:,1]

        with torch.no_grad():
            self.fea_bank[index_batch] = output_f_norm.detach().clone()
            self.score_bank[index_batch] = score_test.detach().clone()
            self.pseudo_bank[index_batch] = pseudo_label.detach().clone()
            distance = output_f_norm @ self.fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=cfg.SFDA.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = self.score_bank[idx_near]

            ## SNC
        rho_batch = torch.histc(pseudo_label, bins=cfg.Dataset.class_num, min=0, max=cfg.Dataset.class_num - 1) / batch
        self.rho = 0.95*self.rho + 0.05*rho_batch       

        softmax_out_un = score_test.unsqueeze(1).expand(
            -1, cfg.SFDA.K, -1
        ).to(self.device)
        
        loss_pos = torch.mean(
            (F.kl_div(softmax_out_un, score_near.to(self.device), reduction="none").sum(-1)).sum(1)
        )
        loss = loss_pos

        mask = torch.ones((batch, batch))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = score_test.T
        dot_neg = score_test @ copy

        dot_neg = ((dot_neg**2) * mask.to(self.device)).sum(-1)
        neg_pred = torch.mean(dot_neg)
        loss_neg = neg_pred * alpha
        loss += loss_neg

        ## IFA
        ratio = cfg.SFDA.lambda_0 * (iter_num / max_iters)
        maxprob,_=torch.max(score_test,dim=1)
        self.Amount, self.Ave, self.Cov = self.update_CV(features_T, pseudo_label, self.Amount, self.Ave, self.Cov)
        loss_ifa_, sigma2 = self.IFA(w, features_T, outputs_T, self.Cov, ratio)
        loss_ifa = cfg.SFDA.alpha_1 * torch.mean(loss_ifa_)
        loss += loss_ifa

        mean_score = torch.stack([torch.mean(self.score_bank[self.pseudo_bank==i], dim=0) for i in range(cfg.Dataset.class_num)])
        cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(cfg.Dataset.class_num).to(self.device))
        Cov1 = self.Cov.view(cfg.Dataset.class_num,-1).unsqueeze(1)
        Cov0 = self.Cov.view(cfg.Dataset.class_num,-1).unsqueeze(0)
        cov_distance = 1 - torch.sum((Cov1*Cov0),dim=2) / (torch.norm(Cov1, dim=2) * torch.norm(Cov0, dim=2) + 1e-12)
        loss_fd = -torch.sum(cov_distance * cov_weight.to(self.device).detach()) / 2
        loss += cfg.SFDA.alpha_2 * loss_fd

        self.Loss_IFA.update(loss_ifa.item())
        self.Loss_POS.update(loss_pos.item())
        self.Loss_NEG.update(loss_neg.item())
        self.Loss_FD.update(loss_fd.item())
        self.Loss_FINAL.update(loss.item())


        return loss

    
    def update_CV(self, features, labels, Amount, Ave, Cov):
        cfg = self.cfg
        N = features.size(0)
        C = cfg.Dataset.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A) # mask

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot) # masking

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (Ave - ave_CxA).view(C, A, 1),
                (Ave - ave_CxA).view(C, 1, A)
            )
        )

        Cov = (Cov.mul(1 - weight_CV).detach() + var_temp.mul(weight_CV)) + additional_CV
        Ave = (Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        Amount = Amount + onehot.sum(0)
        return Amount, Ave, Cov

    def IFA(self, w, features, logit, cv_matrix, ratio):
        N = features.size(0)
        cfg = self.cfg
        C = cfg.Dataset.class_num
        A = features.size(1)
        log_prob_ifa_ = []
        sigma2_ = []
        pseudo_labels = torch.argmax(logit, dim=1).detach()
        for i in range(C):
            labels = (torch.ones(N)*i).to(self.device).long()
            NxW_ij = w.expand(N, C, A)
            NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
            CV_temp = cv_matrix[pseudo_labels]

            sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij-NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
            with torch.no_grad():
                sigma2_.append(torch.mean(sigma2))
            sigma2 = sigma2.mul(torch.eye(C).to(self.device).expand(N, C, C)).sum(2).view(N, C)
            ifa_logit = logit + 0.5 * sigma2
            log_prob_ifa_.append(F.cross_entropy(ifa_logit, labels, reduction='none'))
        log_prob_ifa = torch.stack(log_prob_ifa_)
        loss = torch.sum(2 * log_prob_ifa.T, dim=1)
        return loss, torch.stack(sigma2_)


    def meter_reset(self):
        self.Loss_FINAL.reset()
        self.Loss_POS.reset()
        self.Loss_NEG.reset()
        self.Loss_FD.reset()
        self.Loss_IFA.reset()

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
                        'loss_pos': self.Loss_POS.avg,
                        'loss_neg': self.Loss_NEG.avg,
                        'loss_fd': self.Loss_FD.avg,
                        'loss_ifa': self.Loss_IFA.avg,
                        'train_memory': train_memory,
                        'train_time': train_time,
                        'lr': self.lr
                        }
            self._wandb_log(data=wandb_data, step=epoch_idx+1)

        if cfg.logging:
            self.log.info(
                        f'Seed:{cfg.seed}, '
                        f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                        f'Epoch:{epoch_idx+1}/{cfg.Training.tar_epoch},'
                        f'Acc_t:{self.acc_t:.4f},'
                        f'Acc_s:{self.acc_t_src:.4f},'
                        f'Train_Memory:{train_memory:.4f}MB,'
                        f'Train_Time:{train_time:.4f}s'
                        )
        
        return 