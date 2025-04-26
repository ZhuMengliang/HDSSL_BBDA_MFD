from exp.exp_basic import Exp_Basic
from omegaconf import DictConfig
import wandb, logging, time
from pathlib import Path
from data_provider.data_factory import data_provider, ForeverDataIterator
import sys,torch
from omegaconf import DictConfig, OmegaConf
from utils.meters import AverageMeter
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log=logging.getLogger(__name__)

from SFDA_Lib.SHOT import SHOT_Module
from SFDA_Lib.AaD import AaD_Module
from SFDA_Lib.BMD import BMD_Module
from SFDA_Lib.ELR import ELR_Module
from SFDA_Lib.NRC import NRC_Module
from SFDA_Lib.SFAD import SFAD_Module
from SFDA_Lib.SFCA import SFCA_Module
from SFDA_Lib.SFDA2 import SFDA2_Module
from SFDA_Lib.CoWA import CoWA_Module
SFDA_Module_Dict = {
    'SHOT': SHOT_Module,
    'AaD': AaD_Module,
    'BMD': BMD_Module,
    'ELR': ELR_Module,
    'NRC': NRC_Module,
    'SFAD': SFAD_Module,
    'SFCA': SFCA_Module,
    'SFDA2': SFDA2_Module,
    'CoWA': CoWA_Module,
}
class Exp_SFDA(Exp_Basic):
    def __init__(self, cfg: DictConfig, run: wandb.sdk.wandb_run.Run):
        super().__init__(cfg)
        self.run=run
    def _setup(self):
        cfg=self.cfg
        # dataset path
        src_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.TL_task[0])+'.pt')
        tar_file_path = Path(cfg.Dataset.name+'_'+str(cfg.Dataset.class_num)+'_'+
                        str(cfg.Dataset.seq_len)+'_C'+str(cfg.TL_task[1])+'.pt')
        src_data_path = Path(cfg.Dataset.root_path)/src_file_path
        tar_data_path = Path(cfg.Dataset.root_path)/tar_file_path
        # pre-trained source model path
        self.Src_Model_Path = Path(cfg.Src_Model_Path)/cfg.Dataset.name/(str(cfg.TL_task)+'_Task')
        Path(self.Src_Model_Path).mkdir(parents=True, exist_ok=True)
        self.Src_Model_Name = str(cfg.seed) + '_' + cfg.Model_src.name + \
                    '_I_'+ str(cfg.Dataset.input_format)+'_B_'+str(cfg.Model_src.bottleneck_type) + '.pt'
        # data loader for source model training, target model training and test
        self.tar_train_set, self.tar_train_loader = data_provider(tar_data_path, cfg, flag='train')
        self.tar_test_set, self.tar_test_loader = data_provider(tar_data_path, cfg, flag='test')
        self.src_test_set, self.src_test_loader = data_provider(src_data_path, cfg, flag='test')
        return 
    def _get_optimizer(self, model, flag='src'):
        pass 
    def _get_model(self, flag='src'):
        cfg = self.cfg
        assert flag in ['src', 'tar'], "the flag is wrong"
        args = cfg.Model_src if flag == 'src' else cfg.Model_tar
        model = self.model_dict[args.name].Model(args).float().to(self.device)
        return model
    def SFDA(self):
        cfg = self.cfg
        device = self.device
        model_tar = self._get_model(flag='tar')
        model_src = self._get_model(flag='src')
        try:
            model_src.load_state_dict(torch.load(Path(self.Src_Model_Path)/self.Src_Model_Name))
            model_tar.load_state_dict(torch.load(Path(self.Src_Model_Path)/self.Src_Model_Name))
        except:
            raise FileNotFoundError("The pre-trained source model is not found!")
        SFDA_Module = SFDA_Module_Dict[cfg.SFDA.name]
        SFDA_module = SFDA_Module(cfg=cfg, log=log, run=self.run, model_tar=model_tar, model_src=model_src,
                                tar_test_loader=self.tar_test_loader, src_test_loader=self.src_test_loader)
        # set up
        SFDA_module._setup()
        train_target_iter = ForeverDataIterator(self.tar_train_loader, device=device)
        iters_per_epoch = len(self.tar_train_loader)
        max_iters = iters_per_epoch * cfg.Training.tar_epoch
        TRAIN_MEMORY = AverageMeter('train_memory_footprint', ':.4f')
        Train_Time = AverageMeter('train_time', ':.4f')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        train_memory=0
        for epoch_idx in range(cfg.Training.tar_epoch):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            SFDA_module.meter_reset()
            SFDA_module.model_tar.train()  
            for iter_idx in range(iters_per_epoch):
                X_T, Y_T, index_batch = next(train_target_iter)[0:3]
                X_T, Y_T = X_T.to(device), Y_T.to(device)
                outputs_T, features_T = SFDA_module.model_tar(X_T)
                loss_final = SFDA_module(X_T, features_T, outputs_T, index_batch, 
                                        iter_idx, max_iters, epoch_idx)
                SFDA_module.optimizer.zero_grad()
                loss_final.backward()
                SFDA_module.optimizer.step()
            '''
            evaluate target model
            '''
            if device.type == 'cuda':
                torch.cuda.synchronize()
                train_memory = torch.cuda.max_memory_allocated(device) / 1024**2  
            end_time = time.perf_counter()
            Train_Time.update(end_time - start_time)
            TRAIN_MEMORY.update(train_memory)
            train_memory = TRAIN_MEMORY.avg
            train_time = Train_Time.avg
            SFDA_module.epoch_evaluation(epoch_idx, train_memory, train_time)
            if SFDA_module.scheduler is not None:
                SFDA_module.scheduler.step()
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        assert isinstance(cfg_dict, dict), "配置必须为字典类型"
        log.info(f"For the UDA {cfg_dict['UDA']}")
        log.info(f"For the Training {cfg_dict['Training']}")
        log.info(f"For the Dataset {cfg_dict['Dataset']}")
        log.info(f"For the Optimization {cfg_dict['Opt_src']}")
        log.info(f"For the Model_src {cfg_dict['Model_src']}")    
        return 




