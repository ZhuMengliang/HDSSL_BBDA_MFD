import hydra, wandb, warnings, time
from omegaconf import DictConfig, OmegaConf, open_dict
warnings.filterwarnings("ignore")
from exp.exp_MSBBDA import Exp_MSBBDA # type: ignore
from utils.tools import seed_everything

@hydra.main(version_base=None, config_path='./Configs_TL', config_name='defaults')
def main(cfg:DictConfig):
    TL_task = list(cfg.Dataset.TL_task)
    from itertools import combinations
    Target_task_list = list(combinations(cfg.Dataset.TL_task, 1))
    MSTL_task_list = [
        [[source_domain for source_domain in TL_task if source_domain not in Target_task ], list(Target_task) ]
        for Target_task in Target_task_list
    ]

    # [ [[0,1], [2]] [[0,2], [1]] [[1,2], [0]] ] 是一个大数组
    #train_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    #seed_list=list(range(2000, 2101))
    seed_list = cfg.Training.seed_list
    for seed in seed_list:
        for MSTL_task in MSTL_task_list:
            seed_everything(seed)
            '''
            modify cfg information
            '''
            with open_dict(cfg):
                cfg.Model_tar.class_num=cfg.Dataset.class_num
                cfg.Model_tar.seq_len=cfg.Dataset.seq_len
                cfg.Model_src.class_num=cfg.Dataset.class_num
                cfg.Model_src.seq_len=cfg.Dataset.seq_len
                cfg.MSTL_task=MSTL_task
                cfg.seed=seed
                if cfg.MSBBDA.SMT:
                    cfg.Training.drop_last=True
                    #因为会存在，在最后一个iteration中，batch=1，而引发BN层的错误
                else:
                    cfg.Training.drop_last=False
                assert cfg.Model_src.bottleneck_type == cfg.Model_tar.bottleneck_type, 'cross-model bottleneck layer mismatch'
                
                cfg.Opt_tar.train_epoch=cfg.Training.tar_epoch
                cfg.Opt_tar.weight_decay=cfg.Opt_tar.lr/10
                cfg.Opt_src.train_epoch=cfg.Training.src_epoch
                cfg.Opt_src.weight_decay=cfg.Opt_src.lr/10
                cfg.Wandb.setup.name = str(MSTL_task[0])+'->' + str(MSTL_task[1]) + str(cfg.seed)
                cfg.Wandb.setup.project = Exp_MSBBDA.__name__ + '_new_' + cfg.Dataset.name

                '''
                group_name = train_time + '_' + str(cfg.Model_tar.name) + '_B' +str(cfg.Model_tar.bottleneck_type) \
                            + '_ema_' + str(cfg.BBDA.dkd_ema) + '_N_' + str(cfg.BBDA.N)\
                            + '_M_' + str(cfg.BBDA.M) + '_W_' + str(cfg.BBDA.warm_up_epoch)
                
                INFOR = '_' + '_memall_' + str(cfg.BBDA.mem_all_updated)
                group_name += INFOR
                cfg.Wandb.setup.group = group_name               
                '''
            wandb_cfg=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            assert isinstance(wandb_cfg, dict), "wandb_cfg should be a dict"
            run=wandb.init(config=wandb_cfg, **cfg.Wandb.setup) # type: ignore
            with run:
                exp=Exp_MSBBDA(cfg,run)
                exp._setup()
                if cfg.MSBBDA.SMT and cfg.MSBBDA.kind==1:
                    exp.MS_SMT()
                    #当kind=1时，是将多个源域混合在一起训练单个目标域模型
                else:
                    exp.MS_DFSSL()

                
            
            print(f"For the Training {wandb_cfg["Training"]}")
            print(f"For the Dataset {wandb_cfg["Dataset"]}")
            print(f"For the Optimization {wandb_cfg['Opt_tar']}")
            print(f"For the Model_src {wandb_cfg['Model_src']}")
            print(f"For the Model_tar {wandb_cfg['Model_tar']}")

if __name__ == "__main__":
    main()