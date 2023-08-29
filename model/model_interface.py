from model.model_zoo import ModelZoo
from dataloader.dataloader import MyDataLoader
from module import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import sklearn.metrics as metrics


class ModelInterface(ModelZoo):
    def __init__(self, opt):
        super().__init__(opt)

    def module_setting(self):
        # settings for classifiers and their optimizers (separated)
        if self.opt.model_name in ["mmgaitnet"]:
            self.classifier = mmGaitNet(self.opt.num_classes).cuda()
        elif self.opt.model_name in ["srpnet"]:
            self.classifier = SRPNet(input_shape=[self.opt.batch_size, 4, self.opt.num_pts_per_frame, self.opt.num_frames], num_classes=self.opt.num_classes, feature_transform=True).cuda()
        elif self.opt.model_name in ["tcpcn"]:
            self.classifier = TCPCN(input_shape=[self.opt.batch_size, 5, self.opt.num_pts_per_frame, self.opt.num_frames], num_classes=self.opt.num_classes).cuda()
        elif self.opt.model_name in ["pointnet"]:
            self.classifier = PointNet(self.opt.num_classes).cuda()
        else:
            print("Invalid model. Check --model_name.")
            exit()
        
        self.optimizer_c = torch.optim.Adam(self.classifier.parameters(), lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.opt.decay_rate)

        # if self.opt.model_name in ["mmgaitnet"]:
        #     self.scheduler_c = torch.optim.lr_scheduler.StepLR(self.optimizer_c, step_size=8, gamma=0.1)
        
    def loss_setting(self):
        # loss of classifiers
        if self.opt.model_name in ["mmgaitnet", "pointnet"]:
            self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=self.opt.label_smooth)
        elif self.opt.model_name in ["srpnet", "tcpcn"]:
            pass  # implemented individually in model_zoo.py

    def dataloader_setting(self):
        self.trainDataLoader = DataLoader(MyDataLoader(self.opt,[0,0.8]), batch_size=self.opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.valDataLoader = DataLoader(MyDataLoader(self.opt,[0.8,1]), batch_size=self.opt.batch_size, shuffle=False, num_workers=2,drop_last=True)

        self.log_string("The number of train data is: %d" % len(self.trainDataLoader.dataset))
        self.log_string("The number of val data is: %d" % len(self.valDataLoader.dataset))

    def train(self):
        # configure module, loss function, and dataloader
        self.module_setting()
        self.loss_setting()
        self.dataloader_setting()
        # training
        for epoch in range(1, self.opt.epoch+1):
            
            self.log_string('Epoch  %d/%s:' % (epoch, self.opt.epoch))
            train_loss, train_counter = 0, 0
            
            self.classifier = self.classifier.train() # activate training
            
            for batch_id, (pc, target) in tqdm(enumerate(self.trainDataLoader, 0), total=len(self.trainDataLoader), mininterval=5, smoothing=0.9):
                train_counter += 1
                # configure input
                target = target.squeeze().cuda().long()
                pc = pc.cuda()
                
                if self.opt.model_name == "mmgaitnet":
                    train_loss += self.mmgaitnet_training(pc, target)
                elif self.opt.model_name == "srpnet":
                    train_loss += self.srpnet_training(pc, target, epoch)
                elif self.opt.model_name == "tcpcn":
                    train_loss += self.tcpcn_training(pc, target, epoch)
                elif self.opt.model_name == "pointnet":
                    train_loss += self.pointnet_training(pc, target)
                else:
                    print("Invalid model. Check --model_name.")
                    exit()
            # pause of training, start evaluating the temporary model 
            train_loss /= train_counter
            print("train loss : ", train_loss)

            with torch.no_grad():
                train_acc = self.eval_one_epoch(self.classifier.eval(), self.trainDataLoader, epoch, "train")
                val_acc = self.eval_one_epoch(self.classifier.eval(), self.valDataLoader, epoch, "val")

            self.log_string("val_acc: %f"% val_acc) 
            self.log_string("train_loss: %f"% train_loss)
            self.writer.add_scalar("val_acc", val_acc, epoch)
            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.write_csvwriter(epoch, train_acc, val_acc, train_loss)
            
            if  val_acc >= self.opt.save_acc:
                self.save_checkpoint(epoch, train_acc, val_acc, self.opt.comment, self.classifier, self.optimizer_c, str(self.opt.log_dir), self.opt.model_name)
            epoch += 1
            # end of pause
        # end of training 
        self.log_string('End of training...')
        self.log_string(self.opt.log_dir)
        
    def eval_one_epoch(self, model, loader, epoch, dataset_type):
        test_pred, test_true= [], []
        val_loss,  val_counter= 0, 0
        classifier = model.eval().cuda()
        cls_criterion = nn.CrossEntropyLoss(label_smoothing=0)

        for i, (pc, target) in enumerate(loader):
                target = target.squeeze().cuda().long()
                pc = pc.cuda()

                if self.opt.model_name in ["mmgaitnet"]:
                    pred, _ = classifier(pc)
                elif self.opt.model_name in ["srpnet"]:
                   pred, _, _, _ = classifier(pc)
                elif self.opt.model_name in ["tcpcn"]:
                   pred = classifier(pc)
                elif self.opt.model_name in ["pointnet"]:
                    pred, _ = classifier(pc)
                pred_choice = pred.data.max(1)[1]
                batch_val_loss = cls_criterion(pred,target)
                test_true.append(target.cpu().numpy())
                test_pred.append(pred_choice.detach().cpu().numpy())
                val_loss += batch_val_loss.detach().cpu().numpy()
                val_counter += 1

        print("val loss : ", val_loss/val_counter)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        self.log_string(dataset_type+"_acc: %f"% test_acc)
        self.writer.add_scalar(dataset_type+"_acc", test_acc, epoch)
        #confusion_matrix(test_pred, test_true)
        self.confusion_matrix_deluxe(test_pred, test_true, epoch, dataset_type, test_acc, True, True)
        return test_acc