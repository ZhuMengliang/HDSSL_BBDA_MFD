import torch.nn as nn

class Bottleneck(nn.Module):
    #可以参考对比学习经典论文里面projection层的相关设计
    def __init__(self, in_num=512, bottleneck_num=128,type=1):
        super(Bottleneck, self).__init__()
        if type==1:
            self.fc = nn.Sequential(nn.Linear(in_num, bottleneck_num), nn.BatchNorm1d(bottleneck_num),
                                nn.ReLU(inplace=True), nn.Dropout())
        elif type==2:
            self.fc = nn.Sequential(nn.Linear(in_num, bottleneck_num),nn.BatchNorm1d(bottleneck_num),
                                    nn.ReLU(inplace=True)) 
        elif type==3:
            self.fc = nn.Sequential(nn.Linear(in_num, bottleneck_num), nn.BatchNorm1d(bottleneck_num),)
        elif type==4:
            self.fc = nn.Sequential(nn.Linear(in_num, bottleneck_num), nn.BatchNorm1d(bottleneck_num),
                                    nn.Dropout())


    def forward(self, x):
        x = self.fc(x)
        return x
    

'''
bottleneck layer 的主要作用就是降维，即把ResNet-18 backbone的512维特征降维到128维
具体可参考 https://github.com/tim-learn/SHOT/issues/14

bottleneck layer的做法常见于UDA方法，
具体可参考Transfer-Learning-Library
https://github.com/thuml/Transfer-Learning-Library/issues/209
tllib/alignment/adda.py bsp.py cdan.py dann.py mcc.py
bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )

tllib/alignment/dan.py
bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout()
        )

tllib/alignment/jan.py mdd.py
bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout()
        )

'''