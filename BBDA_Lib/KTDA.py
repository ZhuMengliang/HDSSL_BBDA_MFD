# Source-Free Black-Box Adaptation for Machine Fault Diagnosis
import torch.nn as nn
import torch
import torch.nn.functional as F


def dkd_loss(logits_student, logits_teacher, target, beta, temperature=1, alpha=1):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.update_batch_stats = flag



def bnm_loss(probs):
    _, s_tgt, _ = torch.svd(probs)
    bnm_loss = -torch.mean(s_tgt)
    return bnm_loss


def Data_Division(loader, model, device, cfg):
    from torch.utils.data import TensorDataset, DataLoader
    model.eval()
    start_test = True
    all_pseudo_label = torch.tensor([]).to(device)
    all_output = torch.tensor([]).to(device)
    all_entropy = torch.tensor([]).to(device)
    all_input = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    
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
                all_pseudo_label = torch.max(outputs, 1)[1].float().to(device)
                all_entropy = entropy(outputs)
                all_input = inputs.float().to(device)
                all_label = labels.float().to(device)
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().to(device)), 0)
                pseudo_label = torch.max(outputs, 1)[1].float().to(device)
                all_pseudo_label = torch.cat((all_pseudo_label, pseudo_label.float().to(device)), 0)
                all_label = torch.cat((all_label, labels.float().to(device)), 0)
                all_entropy = torch.cat((all_entropy, entropy(outputs)), 0)
                all_input = torch.cat((all_input, inputs.float().to(device)), 0)
    
    threshold = torch.mean(all_entropy)
    easy_index = (all_entropy < threshold)
    hard_index = (all_entropy >= threshold)

    all_input_easy = all_input[easy_index]
    all_input_hard = all_input[hard_index]
    all_pseudo_label_easy = all_pseudo_label[easy_index] 
    all_pseudo_label_hard = all_pseudo_label[hard_index]  
    
    ACC_easy = torch.mean((torch.squeeze(all_pseudo_label_easy).float() == all_label[easy_index]).float()).item()*100
    ACC_hard = torch.mean((torch.squeeze(all_pseudo_label_hard).float() == all_label[hard_index]).float()).item()*100
    # 创建数据集（将数据移至CPU）
    dataset_easy = TensorDataset(
        all_input_easy.cpu(),  # 输入数据
        all_pseudo_label_easy.cpu(),   # 对应标签
        all_label[easy_index].cpu()
    )
    dataset_hard = TensorDataset(
        all_input_hard.cpu(),
        all_pseudo_label_hard.cpu(),
        all_label[hard_index].cpu()
    )

    # 创建DataLoader
    dataloader_easy = DataLoader(
        dataset_easy,
        batch_size=cfg.Training.batch_size,
        shuffle=True,    # 是否打乱数据
        drop_last=True,  # 是否丢弃最后一个不完整的batch
        pin_memory=(True if cfg.use_gpu else False)  # 启用内存锁页，加速CPU到GPU传输
    )
    dataloader_hard = DataLoader(
        dataset_hard,
        batch_size=cfg.Training.batch_size,
        shuffle=True,    # 是否打乱数据
        drop_last=True,  # 是否丢弃最后一个不完整的batch
        pin_memory=(True if cfg.use_gpu else False)  # 启用内存锁页，加速CPU到GPU传输
    )

    return dataloader_easy, dataloader_hard, ACC_easy, ACC_hard


def entropy(outputs):
    probs = nn.Softmax(dim=1)(outputs)
    entropy = torch.sum(-probs * torch.log(probs + 1e-8), dim=1)
    return entropy
