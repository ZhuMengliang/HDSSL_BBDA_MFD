from exp.exp_basic import Exp_Basic
from omegaconf import DictConfig
import wandb,logging
from pathlib import Path
from data_provider.data_factory import data_provider, ForeverDataIterator
from utils.model_utils import get_optimizer,get_scheduler
from utils.meters import AverageMeter
import torch.nn as nn
from utils.tools import cal_acc
import sys,torch
from omegaconf import DictConfig, OmegaConf
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log=logging.getLogger(__name__)

class Exp_Src(Exp_Basic):
    def __init__(self, cfg: DictConfig, run: wandb.sdk.wandb_run.Run): # type: ignore
        super().__init__(cfg)
        self.run=run
    
    def _setup(self):
        cfg=self.cfg
        src_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.TL_task[0])+'.pt')
        tar_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.TL_task[1])+'.pt')
        src_data_path = Path(cfg.Dataset.root_path)/src_file_path
        tar_data_path = Path(cfg.Dataset.root_path)/tar_file_path
        
        self.Src_Model_Path = Path(cfg.Src_Model_Path)/cfg.Dataset.name/(str(cfg.TL_task)+'_Task')
        Path(self.Src_Model_Path).mkdir(parents=True, exist_ok=True)
        # if cfg.Dataset.name=='PHM':
        #     self.Src_Model_Name = str(cfg.seed) + '_' + cfg.Model_src.name + \
        #                 cfg.Opt_src.name + '_' + str(cfg.Opt_src.lr) + '.pt'
        # else:
        self.Src_Model_Name = str(cfg.seed) + '_' + cfg.Model_src.name + \
                    '_I_'+ str(cfg.Dataset.input_format)+'_B_'+str(cfg.Model_src.bottleneck_type) + '.pt'

        self.src_data_set, self.src_data_loader=data_provider(src_data_path, cfg, flag='train')
        self.tar_data_set, self.tar_data_loader=data_provider(tar_data_path, cfg, flag='train')
        self.test_data_set, self.test_data_loader=data_provider(tar_data_path, cfg, flag='test')

        return   # Implement your setup code here


    def _get_optimizer(self, model,flag='src'):
        cfg=self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args=cfg.Opt_src if flag=='src' else cfg.Opt_tar        
        optimizer=get_optimizer(args, model)
        scheduler=get_scheduler(args, optimizer)
        #还少个scheduler
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

    def SMT(self):
        # source model training 
        cfg=self.cfg
        model_src=self._get_model(flag='src')
        optimizer,scheduler=self._get_optimizer(model=model_src,flag='src')
        train_source_iter=ForeverDataIterator(self.src_data_loader, device=self.device)
        cls_function=self._cls_function()
        LOSS_CLS = AverageMeter('loss_cls', ':.4f')
        iters_per_epoch = len(self.src_data_loader)
        for epoch_idx in range(cfg.Training.src_epoch):
            LOSS_CLS.reset()  # Reset loss meter for this epoch
            model_src.train()  # Set model to training mode
            for iter_idx in range(iters_per_epoch):
                X_S, Y_S = next(train_source_iter)[0:2]
                X_S, Y_S = X_S.to(self.device), Y_S.to(self.device)
                batch=X_S.size(0)
                Y_pre,features=model_src(X_S)
                #Y_pre,features=model.pretrain(X_S)
                loss_cls=cls_function(Y_pre, Y_S)
                LOSS_CLS.update(loss_cls.item(), batch)
                optimizer.zero_grad()
                loss_cls.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()
            '''
            '''
            # evaluate source model
            model_src.eval()
            acc_s=cal_acc(self.src_data_loader, model_src, self.device)[0]
            acc_t=cal_acc(self.test_data_loader, model_src, self.device)[0]
            if cfg.wandb_log:
                wandb_data={
                            'loss_cls': LOSS_CLS.avg,
                            'acc_s': acc_s,
                            'acc_t': acc_t 
                            }
                self._wandb_log(data=wandb_data, step=epoch_idx+1)
            if cfg.logging:
                log.info(f'Seed: {cfg.seed}, '
                        f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
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
        log.info(f"For the Dataset {cfg_dict['Dataset']}")
        log.info(f"For the Optimization {cfg_dict['Opt_src']}")
        log.info(f"For the Model_src {cfg_dict['Model_src']}")    

        return 
    
    



    
