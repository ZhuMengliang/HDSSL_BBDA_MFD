import hydra, wandb, warnings, time
from omegaconf import DictConfig, OmegaConf, open_dict
warnings.filterwarnings("ignore")
from exp.exp_SFDA import Exp_SFDA # type: ignore
from utils.tools import seed_everything

@hydra.main(version_base=None, config_path='./Configs_TL', config_name='defaults')
def main(cfg:DictConfig):
    from itertools import permutations
    TL_task_list = list(permutations(cfg.Dataset.TL_task, 2))
    train_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    #seed_list = list(range(2000, 2101)) + list(range(0, 101))
    #seed_list = cfg.Dataset.seed_list
    seed_list = cfg.Training.seed_list
    #for seed in cfg.Training.seed_list:
    for seed in seed_list:
        for TL_task in TL_task_list:
            seed_everything(seed)
            '''
            modify cfg information
            '''
            with open_dict(cfg):
                cfg.Model_tar.class_num=cfg.Dataset.class_num
                cfg.Model_tar.seq_len=cfg.Dataset.seq_len
                cfg.Model_src.class_num=cfg.Dataset.class_num
                cfg.Model_src.seq_len=cfg.Dataset.seq_len
                cfg.TL_task=TL_task
                cfg.seed=seed
                cfg.Opt_tar.train_epoch=cfg.Training.tar_epoch
                cfg.Opt_tar.weight_decay=cfg.Opt_tar.lr/10
                cfg.train_time=train_time
                cfg.Wandb.setup.name = str(TL_task)+str(cfg.seed)
                cfg.Wandb.setup.project = Exp_SFDA.__name__ + '_new_' + cfg.Dataset.name
                # import numpy as np
                # lr_str='1e'+ str(int(np.log10(cfg.Opt_src.lr)))
                # group_name = train_time + '_' + str(cfg.Model_src.name) + '_' + str(cfg.Opt_src.name)\
                #             +lr_str
                # INFO='_I_'+str(cfg.Dataset.input_format)+'_B_'+str(cfg.Model_src.bottleneck_type)
                # group_name+=INFO
                # cfg.Wandb.setup.group = group_name

                    
            wandb_cfg=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            assert isinstance(wandb_cfg, dict), "wandb_cfg should be a dict"
            run=wandb.init(config=wandb_cfg, **cfg.Wandb.setup) # type: ignore
            with run:
                exp=Exp_SFDA(cfg, run)
                exp._setup()
                exp.SFDA()
            
            print(f"For the Training {wandb_cfg['Training']}")
            print(f"For the Dataset {wandb_cfg['Dataset']}")
            print(f"For the Optimization {wandb_cfg['Opt_src']}")
            print(f"For the Model_src {wandb_cfg['Model_src']}")

if __name__ == "__main__":
    main()