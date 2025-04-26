import torch
from torch.utils.data import Dataset
class Dataset_MFD(Dataset):
    def __init__(self, data_path, input_format='amp0', 
                 scale_norm=False, instance_norm=True,
                 **kwargs):
        super().__init__()
        assert data_path.exists(), "this data file not exists" 
        assert input_format in ['time','amp0','amp1','phase'], "the input format not exists"
        MFD_data=torch.load(data_path)
        self.X=MFD_data[input_format]
        self.Y=MFD_data['label']
        self.len=self.Y.size(0)
        self.label_classes, self.label_counts = torch.unique(self.Y, return_counts=True)
        if scale_norm:
            self.scale_mean = torch.mean(self.X, dim=(0, 2), keepdim=True)
            self.scale_std = torch.std(self.X, dim=(0, 2), keepdim=True, unbiased=False)
            self.X = (self.X-self.scale_mean)/self.scale_std
        if instance_norm:
            self.instance_mean = torch.mean(self.X, dim=2, keepdim=True)
            self.instance_std = torch.std(self.X, dim=2, keepdim=True, unbiased=False) # 注意，得按照有偏方差计算，即分母直接除以样本总数N
            self.X = (self.X-self.instance_mean)/self.instance_std
    def __getitem__(self, index):
        x=self.X[index]
        y=self.Y[index]
        return x,y,index
    def __len__(self):
        return self.len