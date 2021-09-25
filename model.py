import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, bias=False), #out 15*15*12
            #nn.InstanceNorm2d(2 * curdim, affine=True, track_running_stats=True)),  # 看不懂什么意思
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1)  #out 13*13*12

        )
        '''默认stride=kernel_size，不等于的情况下为重叠池化，精度更高
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, bias=False), #out 11*11*24
            # nn.InstanceNorm2d(2 * curdim, affine=True, track_running_stats=True)),  # 看不懂什么意思
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1) #out 10*10*24
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, bias=False),  # out 8*8*48
            # nn.InstanceNorm2d(2 * curdim, affine=True, track_running_stats=True)),  # 看不懂什么意思
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)  # out 7*7*48
        )

        self.full = nn.Sequential(
            nn.Linear(7*7*48,1000),# 2352-1000
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000,100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100,10)
        )

    def forward(self,x):
        x=self.conv1(x)
        #print(x.size())
        x=self.conv2(x)
        x=self.conv3(x)

        x=x.view(x.size(0),-1) #把四维（batchsize，channels,x,y) tensor展开，x.size(0)表示batchsize，-1表示根据batchsize自动分配列数
        x=self.full(x)
        return x



