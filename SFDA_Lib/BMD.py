import torch
import torch.nn as nn
from utils.tools import evaluation
from utils.model_utils import get_optimizer,get_scheduler
from scipy.spatial.distance import cdist
import numpy as np
import torch.nn.functional as F
from utils.meters import AverageMeter
from SFDA_Lib.SFDA_Algorithm import SFDA_Algorithm
#from kmeans_pytorch import kmeans
from fast_pytorch_kmeans import KMeans


class BMD_Module(SFDA_Algorithm):
    def __init__(self, cfg, log, run, model_tar, model_src, tar_test_loader, src_test_loader):
        super(BMD_Module, self).__init__(cfg)
        self.cfg = cfg
        self.log = log 
        self.run = run
        self.model_tar = model_tar
        self.model_src = model_src
        self.tar_test_loader = tar_test_loader
        self.src_test_loader = src_test_loader

        self.Loss_IM = AverageMeter('loss_im', ':.4f')
        self.Loss_ST = AverageMeter('loss_st', ':.4f')
        self.Loss_DYM = AverageMeter('loss_dym', ':.4f')
        self.Loss_FINAL = AverageMeter('loss_final', ':.4f')


    def _setup(self):
        cfg = self.cfg
        with torch.no_grad():
            self.model_tar.eval()
            self.glob_multi_feat_cent, self.all_psd_label = \
                self.init_multi_cent_psd_label(cfg, self.model_tar, self.tar_test_loader)
            
        cfg = self.cfg
        # frozen the classifier module of the target model
        for k, v in self.model_tar.classifier.named_parameters():
            v.requires_grad = False
        self.optimizer, self.scheduler = self._get_optimizer(model=self.model_tar, flag='tar')

    def _wandb_log(self,data,step):
            self.run.log(data=data, step=step)
        
    def _get_optimizer(self, model, flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Opt_src if flag=='src' else cfg.Opt_tar        
        optimizer=get_optimizer(args, model)
        scheduler=get_scheduler(args, optimizer)
        return optimizer, scheduler


    def init_multi_cent_psd_label(self, cfg, model_tar, tar_test_loader ):
        model_tar.eval()
        start_test = True
        all_label = torch.tensor([]).to(self.device)
        all_output = torch.tensor([]).to(self.device)
        all_feature = torch.tensor([]).to(self.device)
        with torch.no_grad():
            iter_test = iter(tar_test_loader) 
            for i in range(len(tar_test_loader)):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1].to(self.device)
                inputs = inputs.to(self.device)
                outputs, features = model_tar(inputs)
                if start_test:
                    all_output = outputs.float().to(self.device)
                    all_label = labels.float().to(self.device)
                    all_feature = features.float().to(self.device)
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().to(self.device)), 0)
                    all_label = torch.cat((all_label, labels.float().to(self.device)), 0)
                    all_feature = torch.cat((all_feature, features.float().to(self.device)), 0)

        all_feature= F.normalize(all_feature, p=2, dim=1)
        topk_num = max(all_feature.shape[0] // (cfg.Dataset.class_num * cfg.SFDA.topk_seg), 1)
        _, all_psd_label = torch.max(all_output, dim=1)
        acc = torch.sum(all_label == all_psd_label) / len(all_label)
        acc_list = [acc]
        multi_cent_num = cfg.SFDA.multi_cent_num
        feat_multi_cent = torch.zeros((cfg.Dataset.class_num, multi_cent_num, cfg.Model_tar.bottleneck_dim)).to(self.device)


        kmeans = KMeans(n_clusters=multi_cent_num, mode='cosine', verbose=0)
        iter_nums = 2
        for iter_idx in range(iter_nums):
            for cls_idx in range(cfg.Dataset.class_num):
                if iter_idx == 0:
                    # We apply TOP-K-Sampling strategy to obtain class balanced feat_cent initialization.
                    feat_samp_idx = torch.topk(all_output[:, cls_idx], topk_num)[1]
                else:
                    # After the first iteration, we make use of the psd_label to construct feat cent.
                    # feat_samp_idx = (all_psd_label == cls_idx)
                    feat_samp_idx = torch.topk(feat_dist[:, cls_idx], topk_num)[1] # type: ignore
                    
                feat_cls_sample = all_feature[feat_samp_idx, :]
                # k-means
                cluster_ids_x = kmeans.fit_predict(feat_cls_sample,)
                cluster_centers = kmeans.centroids
                feat_multi_cent[cls_idx, :] = cluster_centers # type: ignore

            feat_dist = torch.einsum("cmk, nk -> ncm", feat_multi_cent, all_feature) #[N,C,M]
            feat_dist, _ = torch.max(feat_dist, dim=2)  # [N, C]
            feat_dist = torch.softmax(feat_dist, dim=1) # [N, C]
                
            _, all_psd_label = torch.max(feat_dist, dim=1)
            acc = torch.sum(all_psd_label == all_label) / len(all_label)
            acc_list.append(acc)
        return feat_multi_cent, all_psd_label

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


    def forward(self, X_T, features_T, outputs_T, index_batch, iter_num, max_iters, epoch_idx, *args):
        cfg = self.cfg
        
        probs_T = F.softmax(outputs_T, dim=1)
        loss_im = self._IM_loss(outputs_T)
        if epoch_idx >= 1.0:
            loss_st = F.cross_entropy(outputs_T, self.all_psd_label[index_batch].to(torch.long))
            
            normed_emd_feat = features_T / torch.norm(features_T, p=2, dim=1, keepdim=True)
            dym_feat_simi = torch.einsum("cmd, nd -> ncm", self.glob_multi_feat_cent, normed_emd_feat)
            dym_feat_simi, _ = torch.max(dym_feat_simi, dim=2) #[N, C]
            dym_label = torch.softmax(dym_feat_simi, dim=1)    #[N, C]
            loss_dym = F.cross_entropy(outputs_T, dym_label) + F.cross_entropy(dym_feat_simi, probs_T )
            
        else:
            loss_st = torch.tensor(0.0).to(self.device)
            loss_dym = torch.tensor(0.0).to(self.device)

        loss_final = loss_im + cfg.SFDA.alpha*loss_st + cfg.SFDA.beta*loss_dym

        self.Loss_FINAL.update(loss_final.item())
        self.Loss_IM.update(loss_im.item())
        self.Loss_ST.update(loss_st.item())
        self.Loss_DYM.update(loss_dym.item())

        with torch.no_grad():
            self.glob_multi_feat_cent = self.EMA_update_multi_feat_cent_with_feat_simi(
                                    self.glob_multi_feat_cent, features_T, decay=0.9999)

        return loss_final

    def EMA_update_multi_feat_cent_with_feat_simi(self,glob_multi_feat_cent, embed_feat, decay=0.99):
        
        batch_size = embed_feat.shape[0]
        class_num  = glob_multi_feat_cent.shape[0]
        multi_num  = glob_multi_feat_cent.shape[1]
        
        normed_embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_embed_feat)
        feat_simi = feat_simi.flatten(1) #[N, C*M]
        feat_simi = torch.softmax(feat_simi, dim=1).reshape(batch_size, class_num, multi_num) #[N, C, M]
        
        curr_multi_feat_cent = torch.einsum("ncm, nd -> cmd", feat_simi, normed_embed_feat)
        curr_multi_feat_cent /= (torch.sum(feat_simi, dim=0).unsqueeze(2) + 1e-8)
        
        glob_multi_feat_cent = glob_multi_feat_cent * decay + (1 - decay) * curr_multi_feat_cent
        
        return glob_multi_feat_cent



    def meter_reset(self):
        self.Loss_FINAL.reset()

    def epoch_evaluation(self, epoch_idx, train_memory, train_time):
        cfg = self.cfg
        self.model_tar.eval()
        self.acc_t, self.mf1_t, self.P_t, self.features_tar_norm, self.labels, self.predicts, self.features_tar = \
            evaluation(self.tar_test_loader, self.model_tar, self.device)
        self.acc_t_src, self.mf1_t_src = evaluation(self.src_test_loader, self.model_tar, self.device)[0:2]
        self.lr = self.optimizer.param_groups[0]['lr']
        self.glob_multi_feat_cent, self.all_psd_label = \
                self.init_multi_cent_psd_label(cfg, self.model_tar, self.tar_test_loader)

    
        if cfg.wandb_log:
            wandb_data={
                        'acc_t': self.acc_t, 
                        'acc_t_src': self.acc_t_src,
                        'mf1_t': self.mf1_t,
                        'mf1_t_src': self.mf1_t_src,
                        'loss_final': self.Loss_FINAL.avg,
                        'train_memory': train_memory,
                        'train_time': train_time,
                        'lr': self.lr
                        }
            self._wandb_log(data=wandb_data, step=epoch_idx+1)

        if cfg.logging:
            self.log.info(f'Seed: {cfg.seed},'
                    f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                    f'Epoch:{epoch_idx+1}/{cfg.Training.tar_epoch},'
                    f'Acc_t:{self.acc_t:.4f},'
                    f'Acc_s:{self.acc_t_src:.4f},'
                    f'Train_Memory:{train_memory:.4f}MB,'
                    f'Train_Time:{train_time:.4f}s')
        
        return 