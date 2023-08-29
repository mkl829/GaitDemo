# REFERENCE: Meng Z, Fu S, Yan J, et al. Gait recognition for co-existing multiple people using millimeter wave sensing[C]
#                            Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(01): 849-856.
# Re-implementation based on the paper.
import torch
import torch.nn as nn


class mmGaitNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        # PC feature extractor
        self.attributeNet = nn.ModuleList([AttributeNet() for i in range(5)])
        self.squeeze3 = nn.AdaptiveAvgPool2d(1)
        self.excitation3 = nn.Sequential(
            nn.Linear(320, 320 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(320 // 16 ,320),
            nn.Sigmoid()
        )

        self.cbra = nn.Sequential(nn.Conv2d(in_channels=320,out_channels=320,kernel_size=3,stride=1,padding=1,bias=False),
                                  nn.BatchNorm2d(320),
                                  nn.ReLU(inplace=True),
                                  nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.fc_pc = nn.Linear(320, num_classes)
    def forward(self, pc_x):
        # PC
        pc_x = pc_x.transpose(3, 1).contiguous().cuda()
        # pc_x:[batch_size, channels=5, points=128, frames=30] [B C H W]
        splited_x = torch.tensor_split(pc_x,[1,2,3,4],dim=1)  # split on the channels dimension
        # splited_x = torch.tensor_split(x,[1,1,1,1,1],dim=1)  # split on the channels dimension
        output_of_attributeNet_ls = []

        for index, module in enumerate(self.attributeNet):
            output_of_attributeNet_ls.append(module(splited_x[index]))
        # for i in splited_x:
        #     output_of_attributeNet_ls.append(self.attributeNet(i))
        output_of_attributeNet = torch.cat(output_of_attributeNet_ls, dim=1)
        # output_of_attributeNet:[batch_size, 320, 16, 4]
        #
        # squeeze3 = self.squeeze3(output_of_attributeNet)
        # squeeze3 = squeeze3.view(squeeze3.size(0),-1)
        # excitation3 = self.excitation3(squeeze3)
        # excitation3  = excitation3.view(output_of_attributeNet.size(0), output_of_attributeNet.size(1), 1, 1)
        # output_of_attributeNet = output_of_attributeNet * excitation3.expand_as(output_of_attributeNet)
        #
        out = self.cbra(output_of_attributeNet)
        out = torch.flatten(out,1)
        global_feature = out
        out = self.fc_pc(out)
        return out, global_feature

class AttributeNet(nn.Module):
    def __init__(self):
        super(AttributeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)  # padding= (128+2*p-7)/2=64-1? bias=?
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)  # save memory
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)  # save memory
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=2,padding=0,bias=False),
                                        nn.BatchNorm2d(num_features=64))

    def forward(self,x):
        # x: [batch_size, channels=1, points=128, frames=30] [B C H W]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # x: [batch_size, channels=64, 32, 8]
        # ResNet
        identity = x
        out = self.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(x))
        # output: [batch_size, channels=64, 16, 4]
        if self.downsample is not None:
            identity = self.downsample(identity)
            # identity: [batch_size, channels=64, 16, 4]
        out = self.relu(out+identity)
        # output: [batch_size, channels=64, 16, 4]
        ####
        return out
