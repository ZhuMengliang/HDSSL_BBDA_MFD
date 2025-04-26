import torch
from utils.tools import evaluation
from utils.model_utils import get_optimizer,get_scheduler
from scipy.spatial.distance import cdist
import numpy as np
import torch.nn.functional as F
from utils.meters import AverageMeter
from SFDA_Lib.SFDA_Algorithm import SFDA_Algorithm
class SHOT_Module(SFDA_Algorithm):
    def __init__(self, cfg, log, run, model_tar, model_src, tar_test_loader, src_test_loader):
        super(SHOT_Module, self).__init__(cfg)
        self.cfg = cfg
        self.log = log 
        self.run = run
        self.model_tar = model_tar
        self.model_src = model_src
        self.tar_test_loader = tar_test_loader
        self.src_test_loader = src_test_loader
        # loss recorder
        self.LOSS_ST = AverageMeter('loss_st', ':.4f')
        self.Loss_IM = AverageMeter('loss_im', ':.4f')
        self.Loss_FINAL = AverageMeter('loss_final', ':.4f')
    def _setup(self):
        self.acc_t0, self.mf1_t0, self.P_t0, self.features_tar_norm0, self.labels, self.predicts0, \
            self.features_tar0 = evaluation(self.tar_test_loader , self.model_tar, self.device)
        cfg = self.cfg
        # frozen the classifier head of the target model
        for k, v in self.model_tar.classifier.named_parameters():
            v.requires_grad = False
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_tar, flag='tar')
        self.mem_label, self.acc_proto= self._obtain_label(self.features_tar0, self.P_t0, self.labels)
        if cfg.wandb_log:
            wandb_data={'acc_t': self.acc_t0, 'acc_proto': self.acc_proto, 'mf1_t': self.mf1_t0}
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
        bs = outputs.size(0)
        probs = F.softmax(outputs, dim=1)
        entropy = -probs * torch.log(probs + 1e-5)
        entropy = torch.sum(entropy, dim=1)
        entropy = torch.mean(entropy)
        probs_ = probs.mean(dim=0)
        diversity = -torch.sum(-probs_ * torch.log(probs_ + 1e-5))
        loss_IM = entropy + diversity
        return loss_IM
    def _obtain_label(self, features, probs, labels):
        features = torch.cat((features, torch.ones(features.size(0), 1).to(self.device)), dim=1) ## N*D
        features = (features.t() / torch.norm(features, p=2, dim=1)).t() # N*D
        features = features.float().cpu().numpy()
        K = probs.size(1)
        N = probs.size(0)
        aff = probs.float().cpu().numpy() # N*K
        initc = aff.transpose().dot(features) # K*N 
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == labels.float().cpu().numpy()) / N
        print("the initial prototypical acc is {:.2f}".format(acc*100))
        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            acc = np.sum(pred_label == labels.float().cpu().numpy()) / N
            print("Round{} prototypical acc is {:.2f}".format(round, acc*100))
        mem_label = torch.from_numpy(pred_label.astype('int')).to(self.device)
        return mem_label, acc*100
    def forward(self, X_T, features_T, outputs_T, index_batch, iter_num, max_iters, epoch_idx, *args):
        cfg = self.cfg
        P_label =self.mem_label[index_batch].to(torch.long)
        loss_st = F.cross_entropy(outputs_T, P_label)
        loss_im = self._IM_loss(outputs_T)
        loss_final = cfg.SFDA.st*loss_st + cfg.SFDA.im*loss_im
        batch = X_T.size(0)
        self.LOSS_ST.update(loss_st.item(), n=batch)
        self.Loss_IM.update(loss_im.item(), n=batch)
        self.Loss_FINAL.update(loss_final.item(), n=batch)
        return loss_final
    def meter_reset(self):
        self.LOSS_ST.reset()
        self.Loss_IM.reset()
        self.Loss_FINAL.reset()
    def epoch_evaluation(self, epoch_idx, train_memory, train_time):
        cfg = self.cfg
        self.model_tar.eval()
        self.acc_t, self.mf1_t, self.P_t, self.features_tar_norm, self.labels, self.predicts, self.features_tar = \
            evaluation(self.tar_test_loader, self.model_tar, self.device)
        self.acc_t_src, self.mf1_t_src = evaluation(self.src_test_loader, self.model_tar, self.device)[0:2]
        self.lr = self.optimizer.param_groups[0]['lr']
        self.mem_label, self.acc_proto= self._obtain_label(self.features_tar, self.P_t, self.labels)
        if cfg.wandb_log:
            wandb_data={
                        'acc_t': self.acc_t, 
                        'mf1_t': self.mf1_t,
                        'acc_proto': self.acc_proto, 
                        'acc_t_src': self.acc_t_src,
                        'mf1_t_src': self.mf1_t_src,
                        'loss_st': self.LOSS_ST.avg,
                        'loss_im': self.Loss_IM.avg,
                        'loss_final': self.Loss_FINAL.avg,
                        'train_memory': train_memory,
                        'train_time': train_time,
                        'lr': self.lr,
                        }
            
            self._wandb_log(data=wandb_data, step=epoch_idx+1)
        if cfg.logging:
            self.log.info(f'Seed: {cfg.seed}, '
                    f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                    f'Epoch:{epoch_idx+1}/{cfg.Training.tar_epoch},'
                    f'Acc_t:{self.acc_t:.4f},'
                    f'Acc_s:{self.acc_t_src:.4f},'
                    f'Acc_pro:{self.acc_proto:.4f}'
                    f'Train_Memory:{train_memory:.4f}MB,'
                    f'Train_Time:{train_time:.4f}s'
                    )
        return 