import torch
import torch.nn as nn
from utils.tools import evaluation
from utils.model_utils import get_optimizer,get_scheduler
import torch.nn.functional as F
from utils.meters import AverageMeter
from SFDA_Lib.SFDA_Algorithm import SFDA_Algorithm
import numpy as np

class CoWA_Module(SFDA_Algorithm):
    def __init__(self, cfg, log, run, model_tar, model_src, tar_test_loader, src_test_loader):
        super(CoWA_Module, self).__init__(cfg)
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
            self.features_tar0, self.coeff = self.evaluation()
        cfg = self.cfg
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_tar, flag='tar')
        
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

    def evaluation(self,):
        self.model_tar.eval()
        start_test = True
        all_label = torch.tensor([]).to(self.device)
        all_output = torch.tensor([]).to(self.device)
        all_feature = torch.tensor([]).to(self.device)
        with torch.no_grad():
            iter_test = iter(self.tar_test_loader)
            for i in range(len(self.tar_test_loader)):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1].to(self.device)
                inputs = inputs.to(self.device)
                outputs, features = self.model_tar(inputs)
                if start_test:
                    all_output = outputs.float().to(self.device)
                    all_label = labels.float().to(self.device)
                    all_feature = features.float().to(self.device)
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().to(self.device)), 0)
                    all_label = torch.cat((all_label, labels.float().to(self.device)), 0)
                    all_feature = torch.cat((all_feature, features.float().to(self.device)), 0)

        all_feature_norm = F.normalize(all_feature, p=2, dim=1)
        all_feature_raw = all_feature

        prob = nn.Softmax(dim=1)(all_output).to(self.device).detach().clone()
        from torchmetrics import F1Score
        num_classes = prob.size(1)
        f1_metric = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(self.device)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.mean((torch.squeeze(predict).float() == all_label).float()).item()*100
        f1_metric.update(predict, all_label)
        MF1 = f1_metric.compute()

        class_num = prob.size(1)
        ent = torch.sum(-all_output * torch.log(all_output + 1e-8), dim=1)
        unknown_weight = 1 - ent / np.log(class_num)

        K = all_output.shape[1]
        aff = all_output.float()
        initc = torch.matmul(aff.t(), (all_feature_norm))
        initc = initc / (1e-8 + aff.sum(dim=0)[:,None])

        uniform = (torch.ones(len(all_feature_norm), class_num)/class_num).to(self.device)
        pi = prob.sum(dim=0)
        mu = torch.matmul(all_output.t(), (all_feature_norm))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((all_feature_norm), pi, mu, uniform)
        gamma=gamma.float()
        pred_label = gamma.argmax(dim=1)
        for round in range(1):
            pi = gamma.sum(dim=0)
            mu = torch.matmul(gamma.t(), (all_feature_norm))
            mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

            zz, gamma = gmm((all_feature_norm), pi, mu, gamma)
            pred_label = gamma.argmax(dim=1)       

        aff = gamma
        sort_zz = zz.sort(dim=1, descending=True)[0]
        zz_sub = sort_zz[:,0] - sort_zz[:,1]
        
        LPG = zz_sub / zz_sub.max()
        PPL = prob.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
        sample_weight = JMDS

        return accuracy, MF1, prob, all_feature_norm, all_label, predict, all_feature_raw, sample_weight

    def forward(self, X_T, features_T, outputs_T, index_batch, iter_num, max_iters, epoch_idx, *args):
        cfg = self.cfg
        probs_T = nn.Softmax(dim=1)(outputs_T)
        batch = probs_T.size(0)
        lam = (torch.from_numpy(np.random.beta(cfg.SFDA.alpha, cfg.SFDA.alpha, batch))).float().to(self.device)
        t_batch = (torch.eye(cfg.Dataset.class_num).to(self.device))[probs_T.argmax(dim=1)]
        shuffle_idx = torch.randperm(batch)
        mixed_x = (lam * X_T.permute(1,2,0) + (1 - lam) * X_T[shuffle_idx].permute(1,2,0)).permute(2,0,1)
        coeff = self.coeff[index_batch]
        mixed_c = lam * coeff + (1 - lam) * coeff[shuffle_idx]
        mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0)).permute(1,0)
        mixed_x.requires_grad_()
        mixed_c.requires_grad_()
        mixed_t.requires_grad_()
        mixed_outputs = self.model_tar(mixed_x)[0]      
        loss = self.KLLoss(mixed_outputs, mixed_t, mixed_c)
        if epoch_idx+1 <= cfg.SFDA.warmup_epoch:
            loss *= 1e-6
        
        self.Loss_FINAL.update(loss.item())
        return loss

    def KLLoss(self, input_, target_, coeff):
        softmax = nn.Softmax(dim=1)(input_)
        kl_loss = (- target_ * torch.log(softmax + 1e-8)).sum(dim=1)
        kl_loss *= coeff
        return kl_loss.mean(dim=0)

    def meter_reset(self):
        self.Loss_FINAL.reset()

    def epoch_evaluation(self, epoch_idx, train_memory, train_time):
        cfg = self.cfg
        self.model_tar.eval()
        self.acc_t, self.mf1_t, _, _, _, _, _, self.coeff = self.evaluation()
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
                        f'Seed:{cfg.seed}, '
                        f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                        f'Epoch:{epoch_idx+1}/{cfg.Training.tar_epoch},'
                        f'Acc_t:{self.acc_t:.4f},'
                        f'Acc_s:{self.acc_t_src:.4f},'
                        f'Train_Memory:{train_memory:.4f}MB,'
                        f'Train_Time:{train_time:.4f}s'
                        )
        
        return 
    
import math
#下面这个代码跟源代码不一样，是经过deepseek修改完善的
#源码会非正定矩阵无法分解的错
def gmm(all_fea, pi, mu, all_output):
    # 数据校验
    assert torch.isfinite(all_fea).all(), "输入数据含 NaN/Inf!"
    assert torch.isfinite(all_output).all(), "输出数据含 NaN/Inf!"
    assert (all_output >= 0).all() and (all_output <= 1).all(), "输出概率应在[0,1]区间"
    
    # 提升数值精度
    all_fea = all_fea.double()
    mu = [m.double() for m in mu]
    all_output = all_output.double()
    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:, i].unsqueeze(dim=-1)
        
        # 计算 Covi，防止除零
        predi_sum = predi.sum() + 1e-10
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / predi_sum
        
        # 初始对角线加载
        Covi += 1e-5 * torch.eye(temp.shape[1], device=Covi.device)
        
        # 动态调整加载
        diag_load = 1e-5
        max_retries = 20
        for _ in range(max_retries):
            try:
                chol = torch.linalg.cholesky(Covi)
                break
            except RuntimeError:
                Covi += diag_load * torch.eye(Covi.shape[1], device=Covi.device)
                diag_load *= 5
        else:
            eigvals = torch.linalg.eigvalsh(Covi)
            min_eig = eigvals[0].item()
            raise RuntimeError(f"无法正定，i={i}, 最小特征值={min_eig:.3e}")
        
        # 协方差收缩估计
        alpha = 0.1
        Covi = (1 - alpha) * Covi + alpha * torch.eye(Covi.shape[1], device=Covi.device)
        
        # 后续计算...
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5 * (Covi.shape[0] * np.log(2 * math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    
    # 后续堆叠和归一化...
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma