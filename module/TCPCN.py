# REFERENCE: Pegoraro J, Rossi M. Real-time people tracking and identification from sparse mm-wave radar point-clouds[J]. 
#                            IEEE Access, 2021, 9: 78504-78520.
# Re-implementation based on the paper.
import torch
import torch.nn as nn
from module.op import SharedMLP


class CausalDilConv(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, bias=False) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, dilation=dilation, bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        # input: batch, in_channels, length
        x = nn.functional.pad(x, (self.left_padding, 0))
        return super().forward(x)


class TCPCN(nn.Module):
    def __init__(self,
                 input_shape,
                 num_classes,
                 dropout: float=0.5,
                 pc_layers: list = [96, 96, 96, 192, 192],
                 tc_dilations: list = [1, 2, 4],
                 tc_layers: list = [32, 64, 128]):
        super().__init__()
        _, channel, pts_num, frame_num = input_shape
        self.channel = channel
        self.pts_num = pts_num
        self.frame_num = frame_num
        # pc block
        self.pc_layer = SharedMLP(input_channel=5,
                                  layer_units=pc_layers,
                                #   activation=nn.ELU(inplace=True)
                                  activation=nn.ReLU(inplace=True))
        # global pool
        # not use inplace operation after an inplace operation
        self.dropout = nn.Dropout(p=dropout)
        # tc block
        tc_layer = []
        assert (len(tc_layers) == len(tc_dilations))
        tc_layer.append(CausalDilConv(pc_layers[-1],
                                      tc_layers[0],
                                      kernel_size=3,
                                      dilation=tc_dilations[0]))
        for ind in range(len(tc_layers) - 1):
            tc_layer.append(CausalDilConv(tc_layers[ind],
                                          tc_layers[ind + 1],
                                          kernel_size=3,
                                          dilation=tc_dilations[1]))
        self.tc_layer = nn.Sequential(*tc_layer)
        self.conv_fc = nn.Conv1d(tc_layers[-1],
                                 num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=1,
                                 bias=False,
                                 dilation=1)
        # global pool

    def forward(self, x):
        x = x.cuda()
        # batch, feat, pts_num=64, frame=30
        batch = x.size()[0]
        # batch, feat, pts_num*frame
        x = x.reshape(batch, self.channel, -1)
        # batch, 192, -1
        x = self.pc_layer(x)
        x = self.dropout(x)
        # batch, 192, pts_num, frame_num
        x = x.reshape(batch, x.size()[1], self.pts_num, self.frame_num)
        # global average pooling
        # batch, 192, (1), frame_num
        x = torch.mean(x, dim=2)
        # batch, 128, frame_num
        x = self.tc_layer(x)
        # batch, num_classes, frame_num
        x = self.conv_fc(x)
        # global average pooling
        # batch, num_classes
        x = torch.mean(x, dim=2)
        return x


if __name__ == "__main__":
    # x = torch.randn(32, 32, 30)
    # causal_conv = CausalDilConv(in_channels=32, out_channels=64, kernel_size=3, dilation=2)
    # causal_x = causal_conv(x)
    # print(causal_x.shape)
    x = torch.randn(32, 5, 64, 30)
    tcpcn = TCPCN(input_shape=[32, 5, 64, 30], num_classes=2)
    y = tcpcn(x)
    print(y[0].shape)
