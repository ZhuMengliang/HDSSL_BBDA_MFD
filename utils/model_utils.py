import torch.optim as optim
import numpy as np
import torch.nn as nn


def get_optimizer(args, model):
    if args.name=='sgd':
        optimizer = optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    elif args.name=='adam':
        optimizer = optim.Adam(model.parameters(), 
                                lr=args.lr,)
                            #   weight_decay=args.weight_decay)
    elif args.name=='adamw':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.name} is not supported")
    
    return optimizer

def get_scheduler(args, optimizer):

    def lambda1(epoch):
        return (1 + epoch / args.train_epoch) ** (-1 / np.log10(2))

    def lambda2(epoch):
        return (1 + 10 *epoch /args.train_epoch) ** (-0.75)
        # 这个是TL论文里面常用的学习率衰减策略，
        # 通常配合SGD，lr=1e-2，weight decay=1e-3使用
    
    scheduler_dict = {
        'None': None,
        'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=args.lr/10),
        'linear': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - 0.9*epoch / args.train_epoch),
        'self1': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1),
        'self2': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2),
        'exp': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99),

    }
    return scheduler_dict.get(args.scheduler, None)


class MultiModel(nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.models = nn.ModuleList(model_list)  # 关键：使用ModuleList注册子模型
    
    def forward(self):
        pass