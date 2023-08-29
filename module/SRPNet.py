# REFERENCE: Cheng Y, Liu Y. Person reidentification based on automotive radar point clouds[J]. 
#                            IEEE Transactions on Geoscience and Remote Sensing, 2021, 60: 1-13.
# Re-implementation based on the paper.
import torch
import torch.nn as nn
#from op import STN3d, STNkd, SharedMLP
from module.op import STN3d, STNkd, SharedMLP


class AttModule(nn.Module):
    def __init__(self, input_shape: int = 10):
        super(AttModule, self).__init__()
        # batch, feat=1024, frame=input_shape
        self.excitation = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Softmax(dim=1),
            nn.Linear(1024, 50),
            nn.Softmax(dim=1),
            nn.Linear(50, input_shape),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # batch, frame, feat=1024
        x = x.permute(0, 2, 1)
        # batch, 1, feat
        weight = torch.mean(x, dim=1, keepdim=True)
        # batch, 10, 1
        weight = self.excitation(weight).transpose(2, 1)
        return x * weight



class SRPNet(nn.Module):

    def __init__(self, input_shape, num_classes, feature_transform=True):
        super(SRPNet, self).__init__()
        _, channel, pts_num, frame_num = input_shape
        self.channel = channel
        self.pts_num = pts_num
        self.frame_num = frame_num

        self.stn = STN3d()
        self.shared_mlp1 = SharedMLP(input_channel=channel, layer_units=[64, 64])
        self.feature_transform = feature_transform
        self.fstn = STNkd()
        self.shared_mlp2 = SharedMLP(input_channel=64, layer_units=[64, 128, 1024])
        self.attention_module = AttModule(frame_num)
        self.bilstm = nn.LSTM(input_size=1024, batch_first=True,
                              hidden_size=128, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x[:,:,:,:4]
        x = x.transpose(3, 1).contiguous().cuda()
        # batch, channel, pts_num, frame_num
        #print("init:",x.shape, self.channel)
        batch = x.size()[0]
        x = x.reshape(batch, self.channel, -1)
        #print("x1:",x.shape)
        # batch, Matrix[3, 3]
        trans = self.stn(x[:, :3])
        # batch, -1, channel
        x = x.transpose(2, 1)
        x_xyz, x_doppler = torch.split(x, [3, 1], dim=2)
        x_xyz = torch.bmm(x_xyz, trans)
        x = torch.cat([x_xyz, x_doppler], dim=2)
        # batch, channel, -1
        x = x.transpose(2, 1)
        # batch, 64, -1
        x = self.shared_mlp1(x)

        if self.feature_transform:
            # batch, Matrix[64, 64]
            trans_feat = self.fstn(x)
            # batch, -1, 64
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            # bathc, 64, -1
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # batch, 1024, -1
        x = self.shared_mlp2(x)
        # batch, 1024,
        #print(batch, self.pts_num, self.frame_num,x.shape)
        #x = x.reshape(batch, 1024, 128, self.frame_num)
        x = x.reshape(batch, 1024, self.pts_num, self.frame_num)
        # batch, 1024, frame_num
        x = torch.max(x, 2)[0]
        # batch, frame_num, 1024
        x = self.attention_module(x)
        x = self.bilstm(x)[0][:, -1, :]
        x = x.view(batch, 2, 128)
        # MeanPool on the last output of 2 directions
        # batch, 128
        x = torch.mean(x, dim=1)
        global_feature = x
        # print(global_feature.shape)
        x = self.fc(x)
        return x, trans, trans_feat, global_feature


if __name__ == "__main__":
    # x = torch.randn(32, 1024, 10)
    # att = AttModule(10)
    # y = att(x)
    # print(y.shape)
    x = torch.randn(64, 4, 128, 30)
    srpNet = SRPNet(input_shape=[64, 4, 128, 30], num_classes=2, feature_transform=True)
    y = srpNet(x)
    print(y[0].shape)
    test_SRPNet = SRPNet(input_shape=[64, 4, 128, 10], num_classes=5, feature_transform=True)
    print(test_SRPNet)
    #summary(test_SRPNet, input_size=(64,4, 128, 30))
    p_num = sum(p.numel() for p in test_SRPNet.parameters() if p.requires_grad)
    print('requires_grad parameters: {:,}'.format(p_num))
