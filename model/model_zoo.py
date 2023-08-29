from model.model_prototype import ModelPrototype
import torch
import torch.nn as nn


class ModelZoo(ModelPrototype):
    def __init__(self, opt):
        super().__init__(opt)
        #self.classifier = None

    def mmgaitnet_training(self, pc, target):
        pred, _ = self.classifier(pc)
        self.optimizer_c.zero_grad()
        clsLoss = self.cls_criterion(pred, target)
        clsLoss.backward(retain_graph=True)
        self.optimizer_c.step()
        # self.scheduler_c.step()
        return clsLoss.detach().cpu().numpy()

    def srpnet_training(self, pc, target, epoch):
        pred, pc_trans, _, pc_feat = self.classifier(pc)
        self.optimizer_c.zero_grad()
        loss = nn.CrossEntropyLoss()
        identity_matrix = torch.eye(pc_trans.size()[-1], device='cuda').repeat(self.opt.batch_size, 1, 1)
        trans_loss = torch.linalg.matrix_norm(identity_matrix - torch.bmm(pc_trans, pc_trans.permute(0, 2, 1))).sum()
        clsLoss = loss(pred, target) + self.opt.decay_rate * trans_loss	
        clsLoss.backward(retain_graph=True)
        self.optimizer_c.step()
        return clsLoss.detach().cpu().numpy()

    def tcpcn_training(self, pc, target, epoch):
        pred = self.classifier(pc)
        self.optimizer_c.zero_grad()
        loss = nn.CrossEntropyLoss()
        clsLoss = loss(pred, target)
        clsLoss.backward(retain_graph=True)
        self.optimizer_c.step()
        return clsLoss.detach().cpu().numpy()

    def pointnet_training(self, pc, target):
        pred, _ = self.classifier(pc)
        self.optimizer_c.zero_grad()
        clsLoss = self.cls_criterion(pred, target)
        clsLoss.backward(retain_graph=True)
        self.optimizer_c.step()
        return clsLoss.detach().cpu().numpy()
        
