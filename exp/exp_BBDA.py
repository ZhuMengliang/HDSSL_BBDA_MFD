from exp.exp_basic import Exp_Basic
from omegaconf import DictConfig
import wandb,logging, time, os, sys, torch
from pathlib import Path
from data_provider.data_factory import data_provider, ForeverDataIterator
from utils.model_utils import get_optimizer,get_scheduler
from utils.meters import AverageMeter
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.tools import evaluation, seed_everything, get_process_memory, release_memory_resources
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log=logging.getLogger(__name__)

class Exp_BBDA(Exp_Basic):
    def __init__(self, cfg: DictConfig, run: wandb.sdk.wandb_run.Run): # type: ignore
        seed_everything(cfg.seed)
        super().__init__(cfg)
        self.run = run
    def _setup(self):
        cfg=self.cfg
        seed_everything(cfg.seed)
        # dataset path
        src_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.TL_task[0])+'.pt')
        src_data_path = Path(cfg.Dataset.root_path)/src_file_path
        tar_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.TL_task[1])+'.pt')
        tar_data_path = Path(cfg.Dataset.root_path)/tar_file_path
        # pre-trained source model path
        self.Src_Model_Path = Path(cfg.Src_Model_Path)/cfg.Dataset.name/(str(cfg.TL_task)+'_Task')
        Path(self.Src_Model_Path).mkdir(parents=True, exist_ok=True)
        self.Src_Model_Name = str(cfg.seed) + '_' + cfg.Model_src.name + \
                '_I_'+ str(cfg.Dataset.input_format)+'_B_'+str(cfg.Model_src.bottleneck_type) + '.pt'
        # data loader for source model training, target model training and test
        self.src_data_set, self.src_data_loader = data_provider(src_data_path, cfg, flag='train')
        self.tar_data_set, self.tar_data_loader = data_provider(tar_data_path, cfg, flag='train')
        self.test_data_set, self.test_data_loader = data_provider(tar_data_path, cfg, flag='test')
        return  
    def _get_optimizer(self, model, flag='src'):
        cfg = self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        Opt_cfg = cfg.Opt_src if flag == 'src' else cfg.Opt_tar        
        optimizer = get_optimizer(Opt_cfg, model)
        scheduler = get_scheduler(Opt_cfg, optimizer)
        return optimizer, scheduler
    def _get_model(self, flag='src'):
        cfg = self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        model_cfg = cfg.Model_src if flag == 'src' else cfg.Model_tar
        model = self.model_dict[model_cfg.name].Model(model_cfg).float().to(self.device)
        return model
    def _cls_function(self):
        cfg = self.cfg
        assert cfg.Training.cls_function in ['CE', 'LS'], "the cls_function is wrong"
        label_smoothing = 0.1 if cfg.Training.cls_function == 'LS' else 0.0
        cls_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return cls_function
    def _wandb_log(self,data,step):
        self.run.log(data=data, step=step)
    def HDSSL(self):
        # Hierarchical Debiased Self-Supervised Learning
        cfg = self.cfg
        device = self.device
        seed_everything(cfg.seed)
        model_src = self._get_model(flag='src')
        model_tar = self._get_model(flag='tar')
        try:
            print(Path(self.Src_Model_Path)/self.Src_Model_Name)
            model_src.load_state_dict(torch.load(Path(self.Src_Model_Path)/self.Src_Model_Name))
        except:
            raise FileNotFoundError("The source model is not found!")
        model_src.eval()
        acc_t_src, mf1_t_src, P_src = evaluation(self.test_data_loader, model_src, device)[0:3]
        if cfg.wandb_log:
            wandb_data={ 'acc_t': acc_t_src, 'mf1_t': mf1_t_src, }
            self._wandb_log(data=wandb_data, step=0)
        from BBDA_Lib.HDSSL import Balanced_P, ACU_Sample_Division
        mem_P = P_src.clone()
        # The category-wise self-normalization
        mem_P = Balanced_P(mem_P, balanced=cfg.BBDA.balanced, temp=cfg.BBDA.temp)
        log.info(f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                f'Src_Model Acc={acc_t_src:.3f};')
        # Establish memory bank 
        mem_features = torch.randn(P_src.size(0),cfg.Model_tar.bottleneck_dim).to(device)
        mem_probs = torch.randn(P_src.size(0),cfg.Dataset.class_num).to(device)
        # Setup
        LOSS_DKD = AverageMeter('loss_dst', ':.4f')
        LOSS_DPST = AverageMeter('loss_dpst', ':.4f')
        LOSS_HNA = AverageMeter('loss_hna', ':.4f')
        LOSS_FINAL = AverageMeter('loss_final', ':.4f')
        TRAIN_MEMORY = AverageMeter('train_memory_footprint', ':.4f')
        Train_Time = AverageMeter('train_time', ':.4f')
        warm_up_flag=False
        index_certain_all = []
        index_uncertain_all = []
        prototypes_norm = torch.randn(cfg.Dataset.class_num, cfg.Model_tar.bottleneck_dim)
        prototypes_raw = torch.rand_like(prototypes_norm)
        # target model training
        optimizer, scheduler = self._get_optimizer(model=model_tar,flag='tar')
        train_tar_iter = ForeverDataIterator(self.tar_data_loader, device=device)
        iters_per_epoch = len(self.tar_data_loader)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        train_memory=0
        for epoch_idx in range(cfg.Training.tar_epoch):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            LOSS_DKD.reset() 
            LOSS_DPST.reset() 
            LOSS_HNA.reset()
            LOSS_FINAL.reset() 
            model_tar.train()  
            for iter_idx in range(iters_per_epoch):
                X_T, Y_T, index_batch = next(train_tar_iter)
                X_T, Y_T = X_T.to(device), Y_T.to(device)
                batch = X_T.size(0)
                outputs, features=model_tar(X_T)
                probs = nn.Softmax(dim=1)(outputs)
                mem_p = mem_P[index_batch,:].clone()
                loss_dkd = nn.CrossEntropyLoss()(outputs, mem_p)
                LOSS_DKD.update(loss_dkd.item(), batch)
                # update the memory bank
                with torch.no_grad():
                    mem_features[index_batch,:] = F.normalize(features, p=2, dim=1).detach().clone()
                    mem_probs[index_batch,:] = probs.detach().clone()
                # the warm-up training stage finishes
                if warm_up_flag:
                    index_certain_batch = index_certain_all[index_batch]
                    index_uncertain_batch = index_uncertain_all[index_batch]
                    batch_certain = int(index_certain_batch.float().sum().cpu().numpy())
                    batch_uncertain = int(index_uncertain_batch.float().sum().cpu().numpy())
                    all_certain_num = int(index_certain_all.float().sum().cpu().numpy()) 
                    all_uncertain_num = int(index_uncertain_all.float().sum().cpu().numpy()) 
                    # for the certain-aware set
                    if batch_certain > 0:
                        '''
                        Debiased Prototypical Self-Training
                        '''
                        # B*D D*K -> B*K
                        similarity = torch.mm(features[index_certain_batch,:], prototypes_norm.t())
                        # pseudo-labels from the debiased prototypes
                        pseudo_labels = torch.max(similarity, dim=1)[1].detach()
                        loss_dpst = nn.CrossEntropyLoss()(outputs[index_certain_batch, :], pseudo_labels)
                    # when index_certain_batch is empty
                    else:
                        loss_dpst = torch.tensor(0.0)
                    LOSS_DPST.update(loss_dpst.item(), n=batch_certain)
                    # for the uncertain-aware set
                    if batch_uncertain > 0 and all_certain_num>cfg.BBDA.neighborhood and all_uncertain_num>cfg.BBDA.neighborhood:
                        '''
                        Hierarchical Neighborhood Adaptation
                        '''
                        features_unc = features[index_uncertain_batch,:]
                        probs_unc = probs[index_uncertain_batch,:]
                        '''
                        Inter-Set Neighborhood Adaptation and Intra-Set Neighborhood Adaptation
                        '''
                        from BBDA_Lib.HDSSL import Neighborhood_Adaptation
                        loss_inter_set_na, loss_intra_set_na = Neighborhood_Adaptation(features_unc, probs_unc,
                            mem_features, mem_probs, index_certain_all, index_uncertain_all, cfg) 
                        '''
                        Regularized Neighborhood Adaptation
                        '''
                        with torch.no_grad():
                            outputs_prototypes = model_tar.classifier(prototypes_raw) #K*D
                            probs_prototypes = nn.Softmax(dim=1)(outputs_prototypes) # K*K
                            probs_prototypes = probs_prototypes.expand(batch_uncertain, -1, -1)  # B*K*K
                        # B*K*K B*K*1 -> B*K*1
                        loss_reg_na = torch.bmm(probs_prototypes, probs_unc.unsqueeze(2)) 
                        loss_reg_na = loss_reg_na.mean(dim=1).squeeze().mean()
                        loss_hna = loss_intra_set_na + loss_inter_set_na + loss_reg_na
                    # when index_uncertain_batch is empty
                    else:
                        loss_hna = torch.tensor(0.0)
                    LOSS_HNA.update(loss_hna.item(), n=batch_uncertain)
                else:
                    loss_dpst = torch.tensor(0.0)
                    loss_hna = torch.tensor(0.0)
                # final loss for optimization
                loss_final = loss_dkd + cfg.BBDA.dpst*loss_dpst + cfg.BBDA.hna*loss_hna
                LOSS_FINAL.update(loss_final.item())
                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()
            '''
            evaluate target model
            '''
            if device.type == 'cuda':
                torch.cuda.synchronize()
                train_memory = torch.cuda.max_memory_allocated(device) / 1024**2  
            end_time = time.perf_counter()
            Train_Time.update(end_time - start_time)
            TRAIN_MEMORY.update(train_memory)
            model_tar.eval()
            acc_t, mf1_t, P_t, features_tar_norm, _, _, features_tar_raw \
                = evaluation(self.test_data_loader, model_tar, device)
            '''
            Update debiased training predictions
            '''
            # The category-wise self-normalization
            mem_P = Balanced_P(mem_P, balanced=cfg.BBDA.balanced, temp=cfg.BBDA.temp)
            # The sample-wise self-refinement
            mem_P = cfg.BBDA.dkd_ema*mem_P + (1-cfg.BBDA.dkd_ema)*P_t
            lr = optimizer.param_groups[0]['lr']
            if epoch_idx+1 >= cfg.BBDA.warm_up_epoch:
                # Warm-up training stage finishes
                warm_up_flag = True
            # Warm-up training stage finishes
            if warm_up_flag:
                '''
                Adaptive Category-Unbiased Sample Division
                Debiased feature prototypes
                '''
                # prototypes_norm is L2 normalized while prototypes_raw is not
                index_certain_all, index_uncertain_all, prototypes_norm, prototypes_raw = \
                    ACU_Sample_Division(P_t, features_tar_norm, features_tar_raw)
                # Update the memory bank
                with torch.no_grad():
                    mem_features = features_tar_norm.clone().detach()
                    mem_probs = P_t.clone().detach()
            # wandb logging
            if cfg.wandb_log:
                wandb_data={
                            'acc_t': acc_t,
                            'mf1_t': mf1_t,
                            'loss_dkd': LOSS_DKD.avg,
                            'loss_dpst': LOSS_DPST.avg,
                            'loss_hna': LOSS_HNA.avg,
                            'loss_final': LOSS_FINAL.avg,
                            'train_memory': TRAIN_MEMORY.avg,
                            'train_time': Train_Time.avg,
                            'lr': lr,
                            }
                self._wandb_log(data=wandb_data, step=epoch_idx+1)
            # output logging
            if cfg.logging:
                log.info(f'Seed:{cfg.seed},'
                        f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                        f'Epoch:{epoch_idx+1}/{cfg.Training.tar_epoch},'
                        f'Acc_t:{acc_t:.4f},'
                        f'MF1_t:{mf1_t:.4f},'
                        f'Acc_s:{acc_t_src:.4f},'
                        f'Train_Memory:{TRAIN_MEMORY.avg:.4f}MB,'
                        f'Train_Time:{Train_Time.avg:.4f}s')
            # scheduler step
            if scheduler is not None:
                scheduler.step()
        return 