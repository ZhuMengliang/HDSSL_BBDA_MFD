import torch
import torch.nn.functional as F
import torch.nn as nn
def Balanced_P(mem_P, balanced=False, temp=1):
    '''
    The category-wise self-normalization
    '''
    if balanced:
        mem_P = (mem_P**temp) / mem_P.sum(dim=0, keepdim=True)
        mem_P = mem_P / mem_P.sum(dim=1, keepdim=True)
    return mem_P
def Neighborhood_Adaptation(features_unc, probs_unc, mem_features, mem_probs,
                    index_certain_all, index_uncertain_all, cfg):
    '''
    Inter-set Neighborhood_Adaptation 
    '''
    batch = features_unc.size()[0]
    with torch.no_grad():
        # The certain-aware memory bank
        mem_features_c = mem_features[index_certain_all,:]
        mem_probs_c = mem_probs[index_certain_all,:]
        features_unc_norm = F.normalize(features_unc, dim=1)
        # Retrieve the N-nearest neighborhood high-confidence samples
        similarity_c = features_unc_norm@mem_features_c.t()
        _,neighborhood_c = torch.topk(similarity_c,dim=-1,
                                largest=True,sorted=True,
                                k=cfg.BBDA.neighborhood) #B*N
        neighborhood_P_c = mem_probs_c[neighborhood_c] #B*N*K
        neighborhood_P_c = neighborhood_P_c.permute(0,2,1) #B*K*N
    # B*1*K B*K*N -> B*1*N -> B*1*1
    loss_inter_set_na =- torch.bmm(probs_unc.unsqueeze(1), neighborhood_P_c)
    loss_inter_set_na = loss_inter_set_na.sum(-1).squeeze().mean()
    '''
    Intra-set Neighborhood_Adaptation 
    '''
    with torch.no_grad():
        # The uncertain-aware memory bank
        mem_features_unc = mem_features[index_uncertain_all,:]
        mem_probs_unc = mem_probs[index_uncertain_all,:]
        similarity_unc = features_unc_norm@mem_features_unc.t()
        # Retrieve the N-nearest neighborhood low-confidence samples
        _, neighborhood_unc = torch.topk(similarity_unc,dim=-1,
                                    largest=True,sorted=True,
                                    k=cfg.BBDA.neighborhood) #B*N
        neighborhood_P_unc = mem_probs_unc[neighborhood_unc] #B*N*K
        neighborhood_P_unc = neighborhood_P_unc.permute(0,2,1) #B*K*N
    # local neighborhood centroid
    neighborhood_P_unc_ = torch.mean(neighborhood_P_unc, dim=2) #B*K
    # entropy value
    entropy = torch.sum(-neighborhood_P_unc_ * torch.log(neighborhood_P_unc_ + 1e-8), dim=1)
    # sample weights based on the entropy values
    num_classes=cfg.Dataset.class_num
    sample_weights = (1e-8 + torch.exp(-entropy / torch.log(torch.tensor(num_classes))))#B*1
    # B*1*K B*K*1 -> B*1*1 -> B*1
    loss_intra_set_na = -torch.bmm(probs_unc.unsqueeze(1), neighborhood_P_unc_.unsqueeze(2)).squeeze()
    # B*1 B*1 -> B*1
    loss_intra_set_na = (loss_intra_set_na*sample_weights).mean()
    return loss_inter_set_na, loss_intra_set_na
def ACU_Sample_Division(P_t, features_tar_norm, features_tar_raw):
    '''
    Adaptive Category-Unbiased Sample Division
    '''
    num_classes = P_t.size()[1]
    # debiased feature prototypes L2 normalized K*D
    prototypes_norm = torch.zeros(num_classes, features_tar_norm.size()[1]).to(features_tar_norm.device)
    # debiased feature prototypes not L2 normalized K*D
    prototypes_raw= torch.zeros_like(prototypes_norm)
    # adaptive prediction confidence threshold
    confidence = P_t.max(dim=1)[0]
    threshold = confidence.mean()
    ## certain-set data budget for each category
    nc = (confidence > threshold).float().sum().item()//num_classes
    index_certain_all = (torch.ones_like(confidence) == 0)
    # the category-wise certain-aware subset 
    for cls in range(num_classes):
        cls_index = torch.topk(P_t[:, cls], int(nc), largest=True, sorted=True)[1]
        index_certain_all[cls_index] = True
        features_norm_cls = features_tar_norm[cls_index, :]
        prototypes_norm[cls, :] = features_norm_cls.mean(dim=0)+1e-15
        features_raw_cls = features_tar_raw[cls_index, :]
        prototypes_raw[cls, :] = features_raw_cls.mean(dim=0)+1e-15
    prototypes_norm = F.normalize(prototypes_norm, dim=1).detach()# K*D
    index_uncertain_all = ~index_certain_all
    return index_certain_all, index_uncertain_all, prototypes_norm, prototypes_raw
