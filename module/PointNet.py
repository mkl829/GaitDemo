# REFERENCE: Qi C R, Su H, Mo K, et al. Pointnet: Deep learning on point sets for 3d classification and segmentation[C]
#                            Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 652-660.
# The original model is modified to be applied on mmwave radar point clouds.
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        input_dim = 5 # remain modificaiton, 10 is just an example
        self.conv1d1 = nn.Sequential(nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1d2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1d3 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.conv1d4 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True))
        #self.maxpool = nn.Max
        self.att1 = nn.Sequential(nn.Linear(1024,64), nn.BatchNorm1d(64),nn.ReLU(inplace=True), 
                                                            nn.Linear(64,1024), nn.Softmax())
        self.att2 = nn.Sequential(nn.Linear(1024,64), nn.BatchNorm1d(64),nn.ReLU(inplace=True), 
                                                            nn.Linear(64,1024), nn.Softmax())
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(p=0.4))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        self.fc3 = nn.Sequential(nn.Linear(256, num_classes))


    def forward(self, points):
        # points.shape = [B, F, N, C], C=9 (x, y, z, v, s, norm_x, norm_y, norm_z, t) a point is invalid if its t is negative, such as -1.
        B, F, N, C = points.shape
        #pts_partial_compact = points.view(B, F//5, N*5, C)  # neighbour frames are compacted by 5 frames
        # centers = getCenters(points)
        # deltas = points - centers  # delta x, delta y, delta z, etc.
        # ha, va, dist = getRelativeInfo(deltas)
        # points_with_auxiliary_info = torch.cat([points, centers, deltas, ha, va, dist], dim = 1)
        points = points.view(B,F*N,C)
        input = points[:,:,:5]
        input = input.permute(0,2,1)
        # out = self.conv1d1(points_with_auxiliary_info)
        out = self.conv1d1(input)
        out = self.conv1d2(out)
        out = self.conv1d3(out)
        out = self.conv1d4(out)
        #print(out.shape)
        maxpool_out = torch.max(out, 2)[0]
        out = maxpool_out
        out = out.view(-1, 1024)
        out = self.fc1(out)
        out = self.fc2(out)
        feat = out
        out = self.fc3(out)
        return out, feat