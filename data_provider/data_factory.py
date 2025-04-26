from data_provider.data_loader import Dataset_MFD
from torch.utils.data import DataLoader, ConcatDataset
data_dict={
    'MFD': Dataset_MFD,
}
def data_provider(data_path, cfg, flag='train'):
    if flag=='test':
        shuffle=False
        drop_last=False
    else:
        shuffle=True
        drop_last=cfg.Training.drop_last
    Data=data_dict['MFD']
    data_set=Data(data_path=data_path,
                input_format=cfg.Dataset.input_format,
                scale_norm=cfg.Dataset.scale_norm,
                instance_norm=cfg.Dataset.instance_norm)
    data_loader=DataLoader(data_set, 
                    batch_size=cfg.Training.batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=cfg.Training.num_workers,
                    pin_memory=(True if cfg.use_gpu else False))
    return data_set, data_loader

def MSdata_provider(data_paths, cfg, flag='train'):
    if flag=='test':
        shuffle=False
        drop_last=False
    else:
        shuffle=True
        drop_last=cfg.Training.drop_last

    Data=data_dict['MFD']
    data_sets = []
    for i in range(len(data_paths)):
        data_set=Data(data_path=data_paths[i],
                    input_format=cfg.Dataset.input_format,
                    scale_norm=cfg.Dataset.scale_norm,
                    instance_norm=cfg.Dataset.instance_norm)
        data_sets.append(data_set)
    MS_Dataset = ConcatDataset(data_sets)
    
    data_loader=DataLoader(MS_Dataset, 
                    batch_size=cfg.Training.batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=cfg.Training.num_workers,
                    pin_memory=(True if cfg.use_gpu else False))
    
    return MS_Dataset, data_loader


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)
    


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)