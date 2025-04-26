from exp.exp_basic import Exp_Basic
from omegaconf import DictConfig
import wandb,logging,sys,torch
from pathlib import Path
from data_provider.data_factory import MSdata_provider, data_provider, ForeverDataIterator
from utils.model_utils import get_optimizer,get_scheduler
from utils.meters import AverageMeter
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import evaluation
from omegaconf import DictConfig, OmegaConf
from utils.tools import seed_everything
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log=logging.getLogger(__name__)

class Exp_MSBBDA(Exp_Basic):
    def __init__(self, cfg: DictConfig, run: wandb.sdk.wandb_run.Run): # type: ignore
        seed_everything(cfg.seed)
        super().__init__(cfg)
        self.run=run
    
    def _setup(self):
        cfg=self.cfg
        seed_everything(cfg.seed)
        src_file_paths = []
        src_data_paths = []
        for i in range(len(cfg.MSTL_task[0])):
            src_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+\
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.MSTL_task[0][i])+'.pt')
            src_file_paths.append(src_file_path)
            src_data_path = Path(cfg.Dataset.root_path)/src_file_path
            src_data_paths.append(src_data_path)
        tar_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.MSTL_task[1][0])+'.pt')
        tar_data_path = Path(cfg.Dataset.root_path)/tar_file_path
        
        if cfg.MSBBDA.kind==0:
            '''
            Distributed MS-BBDA
            Multiple labeled source domains are distributed across different cloud platforms 
            for separate source model training.
            Multiple input-output APIs of the distributed pre-trained black-box source models are accessible. 
            '''
            self.Src_Model_Paths = []
            for i in range(len(cfg.MSTL_task[0])):
                Src_Model_Path = Path(cfg.Src_Model_Path)/cfg.Dataset.name/('[' + str(cfg.MSTL_task[0][i]) + ', ' +str(cfg.MSTL_task[1][0]) + ']_Task')
                Path(Src_Model_Path).mkdir(parents=True, exist_ok=True)
                self.Src_Model_Paths.append(Src_Model_Path)

        elif cfg.MSBBDA.kind==1:
            '''
            Multiple labeled source domains are centralized on a cloud platform for joint source model training.
            Only the input-output API of the centralized pre-trained black-box source model is accessible.
            '''
            temp=''.join(str(x) for x in cfg.MSTL_task[0])
            self.Src_Model_Path = Path(cfg.Src_Model_Path)/cfg.Dataset.name/(temp + '_' + str(cfg.MSTL_task[1]) + '_Task')
            Path(self.Src_Model_Path).mkdir(parents=True, exist_ok=True)
            self.src_data_set, self.src_data_loader=MSdata_provider(src_data_paths, cfg, flag='train')
        
        self.Src_Model_Name = str(cfg.seed) + '_' + cfg.Model_src.name + \
                '_I_'+ str(cfg.Dataset.input_format)+'_B_'+str(cfg.Model_src.bottleneck_type) + '.pt'
        
        self.tar_data_set, self.tar_data_loader=data_provider(tar_data_path, cfg, flag='train')
        self.test_data_set, self.test_data_loader=data_provider(tar_data_path, cfg, flag='test')
        
        return  

    def _get_optimizer(self, model, flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Opt_src if flag=='src' else cfg.Opt_tar        
        optimizer=get_optimizer(args, model)
        scheduler=get_scheduler(args, optimizer)
        return optimizer, scheduler
    
    def _get_model(self, flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Model_src if flag=='src' else cfg.Model_tar
        model=self.model_dict[args.name].Model(args).float().to(self.device)
        return model
    
    def _cls_function(self):
        cfg=self.cfg
        assert cfg.Training.cls_function in ['CE', 'LS'], "the cls_function is wrong"
        label_smoothing=0.1 if cfg.Training.cls_function=='LS' else 0.0
        cls_function=nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return cls_function
    
    def _wandb_log(self,data,step):
        self.run.log(data=data, step=step)

    def MS_SMT(self):
        # source model training 
        cfg=self.cfg
        assert cfg.MSBBDA.kind == 1, "the MSBBDA.kind is wrong"
        model_src=self._get_model(flag='src')
        optimizer,scheduler=self._get_optimizer(model=model_src,flag='src')
        train_source_iter=ForeverDataIterator(self.src_data_loader, device=self.device)
        cls_function=self._cls_function()
        LOSS_CLS = AverageMeter('loss_cls', ':.4f')
        iters_per_epoch = len(self.src_data_loader)
        for epoch_idx in range(cfg.Training.src_epoch):
            LOSS_CLS.reset()  
            model_src.train() 
            for iter_idx in range(iters_per_epoch):
                X_S, Y_S = next(train_source_iter)[0:2]
                X_S, Y_S = X_S.to(self.device), Y_S.to(self.device)
                batch=X_S.size(0)
                Y_pre,features=model_src(X_S)
                loss_cls=cls_function(Y_pre, Y_S)
                LOSS_CLS.update(loss_cls.item(), batch)
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()

            '''
            '''
            # evaluate source model
            model_src.eval()
            acc_s=evaluation(self.src_data_loader, model_src, self.device)[0]
            acc_t=evaluation(self.test_data_loader, model_src, self.device)[0]
            if cfg.wandb_log:
                wandb_data={
                            'loss_cls': LOSS_CLS.avg,
                            'acc_s': acc_s,
                            'acc_t': acc_t 
                            }
                self._wandb_log(data=wandb_data, step=epoch_idx+1)
            if cfg.logging:
                log.info(f'Seed: {cfg.seed}, '
                        f'TL_Task:{cfg.MSTL_task[0]}->{cfg.MSTL_task[1]},'
                        f'Epoch {epoch_idx+1}/{cfg.Training.src_epoch}, '
                        f'Loss_cls: {LOSS_CLS.avg:.4f}, '
                        f'Acc_s: {acc_s:.4f}, '
                        f'Acc_t: {acc_t:.4f}')
            if scheduler is not None:
                scheduler.step()

        if cfg.save_Src_model:
            torch.save(model_src.state_dict(), Path(self.Src_Model_Path)/self.Src_Model_Name)
        cfg_dict=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        assert isinstance(cfg_dict, dict), "配置必须为字典类型"
        log.info(f"For the Dataset {cfg_dict["Dataset"]}")
        log.info(f"For the Optimization {cfg_dict['Opt_src']}")
        log.info(f"For the Model_src {cfg_dict['Model_src']}")    

        return 

    def MS_HDSSL(self):
        # Hierarchical Debiased Self-Supervised Learning
        cfg=self.cfg
        device=self.device
        seed_everything(cfg.seed)
        assert cfg.MSBBDA.kind in [0,1], "the MSBBDA.kind is wrong"

        if cfg.MSBBDA.kind==0:
            model_src_list=[]
            mem_P = 0
            for i in range(len(self.Src_Model_Paths)):
                Src_Model_Path = self.Src_Model_Paths[i]
                model_src=self._get_model(flag='src')
                try:
                    model_src.load_state_dict(torch.load(Path(Src_Model_Path)/self.Src_Model_Name))
                except:
                    raise FileNotFoundError("The source model is not found!")
                model_src.eval()
                model_src_list.append(model_src)
                _, _, P_src, _, all_label = evaluation(self.test_data_loader, model_src, self.device)[0:5]
                mem_P += P_src.clone()
            mem_P = mem_P/len(self.Src_Model_Paths)
            num_classes = cfg.Dataset.class_num
            all_predict_src = torch.argmax(mem_P, dim=1)
            from torchmetrics import Accuracy, F1Score
            f1_metric = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(self.device)
            acc_metric = Accuracy(num_classes=num_classes, task='multiclass').to(self.device)
            f1_metric.update(all_predict_src, all_label)
            acc_metric.update(all_predict_src, all_label)
            mf1_t_src = f1_metric.compute()
            acc_t_src = acc_metric.compute()
            log.info(f'TL_Task:{cfg.MSTL_task[0]}->{cfg.MSTL_task[1]},'
                f'Src_Model Acc={acc_t_src:.3f}%; MF1={mf1_t_src:.3f}%;')
            if cfg.wandb_log:
                wandb_data={ 'acc_t': acc_t_src, 'mf1_t': mf1_t_src }
                self._wandb_log(data=wandb_data, step=0)

        elif cfg.MSBBDA.kind==1:
            model_src=self._get_model(flag='src')
            try:
                model_src.load_state_dict(torch.load(Path(self.Src_Model_Path)/self.Src_Model_Name))
            except:
                raise FileNotFoundError("The source model is not found!")
            acc_t_src, mf1_t_src, P_src, = evaluation(self.test_data_loader, model_src, self.device)[0:3]
            mem_P = P_src.clone()
            log.info(f'TL_Task:{cfg.MSTL_task[0]}->{cfg.MSTL_task[1]},'
                f'Src_Model Acc={acc_t_src:.3f}%; MF1={mf1_t_src:.3f}%;')
            if cfg.wandb_log:
                wandb_data={ 'acc_t': acc_t_src, 'mf1_t': mf1_t_src}
                self._wandb_log(data=wandb_data, step=0)
        else:
            raise NotImplementedError("The MSBBDA kind is not implemented!")

        model_tar=self._get_model(flag='tar')
        from BBDA_Lib.HDSSL import Balanced_P, ACU_Sample_Division
        mem_P = Balanced_P(mem_P, balanced=cfg.BBDA.balanced, temp=cfg.BBDA.temp)
        # Establish memory bank
        mem_features = torch.randn(P_src.size(0),cfg.Model_tar.bottleneck_dim).to(device)
        mem_probs = torch.randn(P_src.size(0),cfg.Dataset.class_num).to(device)
        # Setup
        LOSS_DKD = AverageMeter('loss_dst', ':.4f')
        LOSS_DPST = AverageMeter('loss_dpst', ':.4f')
        LOSS_HNA = AverageMeter('loss_hna', ':.4f')
        LOSS_FINAL = AverageMeter('loss_final', ':.4f')
        warm_up_flag=False
        index_certain_all = []
        index_uncertain_all = []
        prototypes_norm = torch.randn(cfg.Dataset.class_num, cfg.Model_tar.bottleneck_dim)
        prototypes_raw = torch.rand_like(prototypes_norm)
        # target model training
        optimizer, scheduler = self._get_optimizer(model=model_tar,flag='tar')
        train_tar_iter = ForeverDataIterator(self.tar_data_loader, device=device)
        iters_per_epoch = len(self.tar_data_loader)
        for epoch_idx in range(cfg.Training.tar_epoch):
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
                        f'Acc_s:{acc_t_src:.4f},')
            # scheduler step
            if scheduler is not None:
                scheduler.step()
        return 

