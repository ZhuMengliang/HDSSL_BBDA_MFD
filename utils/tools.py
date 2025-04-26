import random, os, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score, Accuracy
import psutil, gc, time
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True 
def evaluation(loader, model, device):
    #overall accuracy, macro F1-score
    model.eval()
    start_test = True
    all_label = torch.tensor([]).to(device)
    all_output = torch.tensor([]).to(device)
    all_feature = torch.tensor([]).to(device)
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1].to(device)
            inputs = inputs.to(device)
            outputs, features = model(inputs)
            if start_test:
                all_output = outputs.float().to(device)
                all_label = labels.long().to(device)
                all_feature = features.float().to(device)
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().to(device)), 0)
                all_label = torch.cat((all_label, labels.long().to(device)), 0)
                all_feature = torch.cat((all_feature, features.float().to(device)), 0)
    all_feature_norm = F.normalize(all_feature, p=2, dim=1)
    all_feature_raw = all_feature
    all_prob = nn.Softmax(dim=1)(all_output).to(device).detach().clone()
    _, all_predict = torch.max(all_output, 1)
    num_classes=all_prob.size(1)
    f1_metric = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(device)
    acc_metric = Accuracy(num_classes=num_classes, task='multiclass').to(device)
    f1_metric.update(all_predict, all_label)
    acc_metric.update(all_predict, all_label)
    MF1 = f1_metric.compute()
    Acc = acc_metric.compute()
    return Acc, MF1, all_prob, all_feature_norm, all_label, all_predict, all_feature_raw

def get_process_memory(PID):
    """获取进程及其子进程的独占内存（USS，单位：MB）"""
    process = psutil.Process(PID)
    total_uss = process.memory_full_info().uss
    for child in process.children(recursive=True):
        total_uss += child.memory_full_info().uss
    return total_uss / (1024 ** 2)
def release_memory_resources(device):
    """释放 PyTorch 缓存并触发垃圾回收"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # 释放 GPU 缓存（如有）
    gc.collect()
    time.sleep(0.5)  # 等待内存稳定

