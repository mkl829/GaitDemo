from datetime import datetime
from tensorboardX import SummaryWriter
import os
import logging
import csv
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import torch


class ModelPrototype:
    def __init__(self, opt):
        self.opt = opt
        self.set_logger()
        self.log_string(opt)
        self.writer = SummaryWriter(logdir=opt.log_dir)
        self.cm_path = os.path.join(opt.log_dir,'confusion_matrix')
        if not os.path.exists(self.cm_path):
            os.makedirs(self.cm_path)
        self.csvfile = open(os.path.join(opt.log_dir,"record.csv"),mode='a',newline='')
        self.csvwriter = csv.writer(self.csvfile)
        self.set_csvwriter()
        self.confusion_matrixfile = open(os.path.join(opt.log_dir,"confusion_matrix.csv"),mode='a',newline='')
        self.cmwriter = csv.writer(self.confusion_matrixfile)
        
    def set_logger(self):
        # initialization of logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.opt.log_dir = os.path.join(self.opt.log_dir, self.opt.model_name, current_time)
        if not os.path.exists(self.opt.log_dir):
            os.makedirs(self.opt.log_dir)
        print('checkpoints:', self.opt.log_dir)
        self.logger = logging.getLogger("LOG")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        file_handler = logging.FileHandler(os.path.join(self.opt.log_dir, "log_train.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def set_csvwriter(self):
        # initialization of csvwriter
        self.csvwriter.writerow(["epoch", "train_acc", "val_acc", "train_loss", self.opt.comment])
        
    def write_csvwriter(self, *args):
        print(args)
        self.csvwriter.writerow(list(args))
        
    def log_string(self, msg):
        print(msg)
        self.logger.info(msg)
    
    def confusion_matrix(self, pred, target, need_print=True):
        # a simplified version, no image is saved
        num_classes = self.opt.num_classes
        classes = [chr(i) for i in range(65,65+num_classes)]  # ASCII(A) = 65
        matrix = np.zeros((num_classes,num_classes))
        for i in range(len(pred)):
            matrix[target[i]][pred[i]] += 1
            #matrix[target[i][0]][pred[i]] += 1
        #print(matrix)
        recall = sum([matrix[i, i]/sum(matrix[:,i]) for i in range(num_classes)])/num_classes
        for i in range(num_classes):
            temp = sum(matrix[i])
            for j in range(num_classes):
                matrix[i][j] /= temp
        #print(matrix)
        macro_avg_acc = sum([matrix[i][i] for i in range(num_classes)])/num_classes  # the value equals to precision since all the subjects have the same amount of samples
        
        f1 = (2 * macro_avg_acc * recall)/(macro_avg_acc+recall)
        matrix = np.around(matrix,3)
        if need_print:
            print(matrix)
            print("avg of accuracy of each class:", macro_avg_acc)
        
        return matrix, classes, macro_avg_acc, recall, f1

    def confusion_matrix_deluxe(self, pred, target, epoch, dataset_type, acc, need_print, need_save_fig):
        # a complete version, image can be saved
        # NOTICE: Save as images may be inconvenient to process, and cost plenty of memory. May modify the code and save the confusion matrices in xxx.csv .
        matrix, classes, macro_avg_acc, recall, f1 = self.confusion_matrix(pred, target, need_print)
        self.cmwriter.writerow([matrix,macro_avg_acc,recall,f1,epoch,dataset_type,acc])
        if not need_save_fig:
            return
        
        plt.imshow(matrix, interpolation="nearest", cmap=plt.get_cmap("Oranges"))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)
    
        iters = np.reshape([[[i,j] for j in range(self.opt.num_classes)] for i in range(self.opt.num_classes)],(matrix.size,2))
        for i, j in iters:
            plt.text(j, i, format(matrix[i, j]), va="center", ha="center")
    
        plt.ylabel("Label")
        plt.xlabel("Prediction")
        plt.title(str(epoch) + '_' + str(dataset_type) + '_' + str(round(acc, 3)) + '_' + str(round(macro_avg_acc, 3)))
        plt.tight_layout()
        plt.savefig(self.cm_path + '/' + str(time.strftime("%Y%m%d_%H%M%S", time.localtime())) + '_' + str(epoch) + '_' + str(dataset_type) + ".jpg", dpi=400)
        plt.close("all")
                
    def save_checkpoint(self, epoch, train_accuracy, val_accuracy, comment, model, optimizer, path, modelnet='checkpoint'):
        savepath = path + '/%s-%f-%04d.pth' % (modelnet, val_accuracy, epoch)
        print(savepath)
        state = {
            'epoch': epoch,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'comment': comment,
            #'model_state_dict': model.module.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
