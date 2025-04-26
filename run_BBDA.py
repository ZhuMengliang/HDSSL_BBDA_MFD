import hydra, wandb, warnings
from omegaconf import DictConfig, OmegaConf, open_dict
warnings.filterwarnings("ignore")
from exp.exp_BBDA import Exp_BBDA 
from utils.tools import seed_everything
@hydra.main(version_base=None, config_path='./Configs_TL', config_name='defaults')
def main(cfg:DictConfig):
    from itertools import permutations
    TL_task_list = list(permutations(cfg.Dataset.TL_task, 2))
    seed_list = cfg.Training.seed_list
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
                assert cfg.Model_src.bottleneck_type == cfg.Model_tar.bottleneck_type, 'cross-model bottleneck layer mismatch'
                cfg.Opt_tar.train_epoch=cfg.Training.tar_epoch
                cfg.Opt_tar.weight_decay=cfg.Opt_tar.lr/10
                cfg.Wandb.setup.name = str(TL_task)+str(cfg.seed)
                cfg.Wandb.setup.project = Exp_BBDA.__name__ + '_new_' + cfg.Dataset.name
                #cfg.Wandb.setup.group = group_name 
            wandb_cfg=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            assert isinstance(wandb_cfg, dict), "wandb_cfg should be a dict"
            run=wandb.init(config=wandb_cfg, **cfg.Wandb.setup) # type: ignore
            with run:
                exp=Exp_BBDA(cfg,run)
                exp._setup()
                if cfg.BBDA.name=='HDSSL':
                    exp.HDSSL()
                elif cfg.BBDA.name=='KTDA':
                    exp.KTDA()
                elif cfg.BBDA.name=='DINE':
                    exp.DINE()
                else:
                    raise ValueError(f'Unknown BBDA name: {cfg.BBDA.name}')   
            print(f"For the Training {wandb_cfg['Training']}")
            print(f"For the Dataset {wandb_cfg['Dataset']}")
            print(f"For the Optimization {wandb_cfg['Opt_tar']}")
            print(f"For the Model_src {wandb_cfg['Model_src']}")
            print(f"For the Model_tar {wandb_cfg['Model_tar']}")
if __name__ == "__main__":
    main()