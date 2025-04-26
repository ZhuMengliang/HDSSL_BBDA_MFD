import torch.nn as nn
import torch
import os
class SFDA_Algorithm(nn.Module):
    def __init__(self, cfg):
        super(SFDA_Algorithm, self).__init__()
        self.cfg = cfg
        self.device=self._get_device()
    def _setup(self):
        raise NotImplementedError
        return None
    def _get_device(self):
        cfg=self.cfg
        if cfg.use_gpu and cfg.gpu_type == 'cuda':
            device = torch.device('cuda:{}'.format(cfg.gpu_id))
            print('Use GPU: cuda:{}'.format(cfg.gpu_id))
        elif cfg.use_gpu and cfg.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    def _get_optimizer(self, *args, **kwargs):
        raise NotImplementedError
        return None, None
    def _acquire_device(self):
        cfg=self.cfg
        if cfg.use_gpu and cfg.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                cfg.gpu_id) if not cfg.use_multi_gpu else cfg.devices
            device = torch.device('cuda:{}'.format(cfg.gpu_id))
            print('Use GPU: cuda:{}'.format(cfg.gpu_id))
        elif cfg.use_gpu and cfg.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device


