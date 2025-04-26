import torch.nn as nn
import torch
try:
    import Bottleneck # for test in this file
except:
    import models.Bottleneck as Bottleneck  # for the global run.py


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, 
                    kernel_size=1,stride=stride, bias=False)


class ResBlock(nn.Module):
    # skip-connection
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg # model cfg
        self.blocks=cfg.blocks #[2,2,2,2] for ResNet18
        assert len(self.blocks)<=4, "the number of blocks should not exceed 4"
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        stride_list=[1,2,2,2]
        planes_list=[64,128,256,512]
        layers=[self._make_layer(block=ResBlock, planes=planes_list[i], 
                                 block_num=self.blocks[i], stride=stride_list[i]) 
                                for i in range(len(self.blocks))]
        self.Res_layers=nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if self.cfg.bottleneck:
            self.bottleneck=Bottleneck.Bottleneck(planes_list[len(self.blocks)-1], self.cfg.bottleneck_dim,
                                            type=self.cfg.bottleneck_type)
            fea_dim=self.cfg.bottleneck_dim
        else:
            self.bottleneck=nn.Identity()
            fea_dim=planes_list[len(self.blocks)-1]
        
        self.classifier = nn.Linear(fea_dim, self.cfg.class_num)

        #初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        assert x.size(1)==1, "the input dim should be 1"
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.Res_layers(x)
        x = self.avgpool(x)
        x= x.flatten(start_dim = 1) # torch.flatten(start_dim=1)
        x = self.bottleneck(x)

        return x
    
    def forward(self, x):
        features = self.forward_features(x)
        out = self.classifier(features)
        return out,features
    
    def __len__(self):
        self.len=1+sum(self.blocks)*2
        if self.cfg.bottleneck:
            self.len+=1
        return self.len

if __name__ == "__main__":
    import Bottleneck
    from types import SimpleNamespace
    cfg = SimpleNamespace()
    cfg.blocks=[2,2,2,2] # backbone 17层，原ResNet18的最后一层全连接层被丢弃
    cfg.bottleneck=True
    cfg.bottleneck_dim=128
    cfg.bottleneck_type=1
    cfg.class_num=10
    model=Model(cfg)
    print(len(model))
    print(model)
    input_data=torch.randn(64,1,2000)
    output,feature=model(input_data)
    print(output.shape) # torch.Size([1, 512])
    print(feature.shape) # torch.Size([1, 128])
    print(model.classifier)


        
    
