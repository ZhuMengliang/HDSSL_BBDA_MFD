from exp.exp_basic import Exp_Basic
from omegaconf import DictConfig
import wandb, logging, sys, torch, time
from pathlib import Path
from data_provider.data_factory import data_provider, ForeverDataIterator
from utils.meters import AverageMeter
from utils.tools import evaluation
from utils.model_utils import get_optimizer,get_scheduler
from omegaconf import DictConfig, OmegaConf
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log=logging.getLogger(__name__)

from UDA_Lib.BNM import BNM_Module
from UDA_Lib.BSP import BSP_Module
from UDA_Lib.CDAN import CDAN_Module
from UDA_Lib.CORAL import CORAL_Module
from UDA_Lib.DAN import DAN_Module
from UDA_Lib.DANN import DANN_Module
from UDA_Lib.JAN import JAN_Module
from UDA_Lib.MCC import MCC_Module
from UDA_Lib.SAFN import SAFN_Module
from UDA_Lib.ATDOC_NA import ATDOC_NA_Module
UDA_Module_Dict = {
    'BNM': BNM_Module,
    'BSP': BSP_Module,
    'CDAN': CDAN_Module,
    'CORAL': CORAL_Module,
    'DAN': DAN_Module,
    'DANN': DANN_Module,
    'JAN': JAN_Module,
    'MCC': MCC_Module,
    'SAFN': SAFN_Module,
    'ATDOC_NA': ATDOC_NA_Module
}
class Exp_UDA(Exp_Basic):
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
        # dropout_last=Trie for src_data_loader and tar_data_loader
        self.src_data_set, self.src_data_loader = data_provider(src_data_path, cfg, flag='train')
        self.tar_data_set, self.tar_data_loader = data_provider(tar_data_path, cfg, flag='train')
        self.test_data_set, self.test_data_loader = data_provider(tar_data_path, cfg, flag='test')
        self.train_data_set, self.train_data_loader = data_provider(src_data_path, cfg, flag='test')
        return   
    def _get_optimizer(self, model, flag='src'):
        cfg = self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args = cfg.Opt_src if flag == 'src' else cfg.Opt_tar        
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        return optimizer, scheduler
    def _get_model(self, flag='src'):
        cfg = self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args = cfg.Model_src if flag == 'src' else cfg.Model_tar
        model = self.model_dict[args.name].Model(args).float().to(self.device)
        return model
    def _wandb_log(self,data,step):
        self.run.log(data=data, step=step)
    def UDA(self):
        cfg = self.cfg
        device = self.device
        UDA_Module = UDA_Module_Dict[cfg.UDA.name]
        model_src = self._get_model(flag='src')
        UDA_module = UDA_Module(cfg=cfg, model_src=model_src, test_data_set=self.test_data_set, 
                                src_data_loader=self.src_data_loader)
        # Setup
        UDA_module._setup()
        LOSS_CLS = AverageMeter('loss_cls', ':.4f')
        Loss_TL = AverageMeter('loss_tl', ':.4f')
        Loss_REG = AverageMeter('loss_reg', ':.4f')
        Loss_FINAL = AverageMeter('loss_final', ':.4f')
        DOMAIN_ACC = AverageMeter('domain_acc', ':.4f')
        TRAIN_MEMORY = AverageMeter('train_memory_footprint', ':.4f')
        Train_Time = AverageMeter('train_time', ':.4f')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        train_memory=0        
        # source model training in UDA
        train_source_iter = ForeverDataIterator(self.src_data_loader, device=device)
        train_target_iter = ForeverDataIterator(self.tar_data_loader, device=device)
        iters_per_epoch = len(self.src_data_loader)
        for epoch_idx in range(cfg.Training.src_epoch):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            Loss_REG.reset()
            DOMAIN_ACC.reset()
            LOSS_CLS.reset()
            Loss_TL.reset()
            Loss_FINAL.reset()
            model_src.train() 
            for iter_idx in range(iters_per_epoch):
                X_S, Y_S, index_S = next(train_source_iter)[0:3]
                X_S, Y_S = X_S.to(device), Y_S.to(device)
                X_T, _,  index_T = next(train_target_iter)[0:3]
                X_T = X_T.to(device)
                batch = X_S.size(0)
                X = torch.cat([X_S, X_T], dim=0)
                outputs, features = model_src(X)
                # forward to obtain the UDA losses
                loss_final, loss_cls, loss_tl, loss_reg, domain_acc =\
                    UDA_module(outputs, features, Y_S, X_S, X_T, index_S, index_T)
                UDA_module.optimizer.zero_grad()
                loss_final.backward()
                UDA_module.optimizer.step()
                # record losses
                LOSS_CLS.update(loss_cls.item(), batch)
                Loss_TL.update(loss_tl.item(), batch)
                Loss_REG.update(loss_reg.item(), batch)
                DOMAIN_ACC.update(domain_acc, 2*batch)
                Loss_FINAL.update(loss_final.item(), batch)
            '''
            evaluate source model
            '''
            if device.type == 'cuda':
                torch.cuda.synchronize()
                train_memory = torch.cuda.max_memory_allocated(device) / 1024**2  
            end_time = time.perf_counter()
            Train_Time.update(end_time - start_time)
            TRAIN_MEMORY.update(train_memory)
            model_src.eval()
            acc_s, mf1_s = evaluation(self.train_data_loader, model_src, device)[0:2]
            acc_t, mf1_t = evaluation(self.test_data_loader, model_src, device)[0:2]
            if cfg.wandb_log:
                wandb_data={
                            'loss_cls': LOSS_CLS.avg,
                            'loss_tl': Loss_TL.avg,
                            'loss_reg': Loss_REG.avg,
                            'domain_acc': DOMAIN_ACC.avg,
                            'loss_final': Loss_FINAL.avg,
                            'acc_s': acc_s,
                            'acc_t': acc_t,
                            'mf1_s': mf1_s,
                            'mf1_t': mf1_t,
                            'train_memory': TRAIN_MEMORY.avg,
                            'train_time': Train_Time.avg,
                            }
                self._wandb_log(data=wandb_data, step=epoch_idx+1)
            if cfg.logging:
                log.info(
                    f'Seed:{cfg.seed},'
                    f'TL_Task:{cfg.TL_task[0]}->{cfg.TL_task[1]},'
                    f'Epoch:{epoch_idx+1}/{cfg.Training.src_epoch},'
                    f'Acc_s:{acc_s:.4f},'
                    f'Acc_t:{acc_t:.4f},'
                    f'Train_Memory:{TRAIN_MEMORY.avg:.4f}MB,'
                    f'Train_Time:{Train_Time.avg:.4f}s'
                        )
            if UDA_module.scheduler is not None:
                UDA_module.scheduler.step()
        
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        assert isinstance(cfg_dict, dict), "配置必须为字典类型"
        log.info(f"For the UDA {cfg_dict['UDA']}")
        log.info(f"For the Training {cfg_dict['Training']}")
        log.info(f"For the Dataset {cfg_dict['Dataset']}")
        log.info(f"For the Optimization {cfg_dict['Opt_src']}")
        log.info(f"For the Model_src {cfg_dict['Model_src']}")    
        return 




