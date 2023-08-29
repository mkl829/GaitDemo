from dataloader.dataloader import MyDataLoader
from torch.utils.data import DataLoader
import torch
import numpy as np
import sklearn.metrics as metrics
import argparse
from module import *
#---- parameter setting ----#
parser = argparse.ArgumentParser()
# model unrelated parameters (path etc.)
parser.add_argument("--data_dir", default="./data/60Ghz_radar/")
parser.add_argument("--pretrained_parameters_path", type=str, help="the path of pretrained parameters (e.g., 'xxx.pth')")
parser.add_argument("--model_name", type=str, help="model name: mmgaitnet/srpnet/tcpcn/pointnet") # used for logging and model setting
# model related hyper parameters
parser.add_argument("--num_classes", type=int, default=5, help="number of classes for dataset")
parser.add_argument("--num_frames", type=int, default=60, help="number of frames for a sample")
parser.add_argument("--batch_size", type=int, help="size of the batches")
parser.add_argument("--num_pts_per_frame", default=64, type=int, help="number of points per frame")
opt = parser.parse_args()
print(opt)
######

def eval(loader, pretrained_parameters_path):
    # Load models
    if opt.model_name in ["mmgaitnet"]:
        classifier = mmGaitNet(opt.num_classes).cuda()
    elif opt.model_name in ["srpnet"]:
        classifier = SRPNet(input_shape=[opt.batch_size, 4, opt.num_pts_per_frame, opt.num_frames], num_classes=opt.num_classes, feature_transform=True).cuda()
    elif opt.model_name in ["tcpcn"]:
        classifier = TCPCN(input_shape=[opt.batch_size, 5, opt.num_pts_per_frame, opt.num_frames], num_classes=opt.num_classes).cuda()
    elif opt.model_name in ["pointnet"]:
        classifier = PointNet(opt.num_classes).cuda()
    else:
        print("Invalid model. Check --model_name.")
        exit()
    
    checkpoint = torch.load(pretrained_parameters_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    test_pred = []
    test_true = []
    
    classifier = classifier.eval().cuda()
    ###################### for threshold
    sample_counter = 0
    correct_counter = 0  # make a correct decision
    wrong_counter = 0  # make a wrong decision
    ignore_counter = 0  # no decision
    
    ####################################
    for i, (pc, target) in enumerate(loader):
        target = target.squeeze().cuda().long()
        if opt.model_name in ["mmgaitnet"]:
            pred, _ = classifier(pc)
        elif opt.model_name in ["srpnet"]:
            pred, _, _, _ = classifier(pc)
        elif opt.model_name in ["tcpcn"]:
            pred = classifier(pc)
        elif opt.model_name in ["pointnet"]:
            pred, _ = classifier(pc)
        pred_choice = pred.data.max(1)[1]
        test_true.append(target.cpu().numpy())
        test_pred.append(pred_choice.detach().cpu().numpy())
        pred_softmax = torch.softmax(pred,dim=1)
        pred_possi = pred_softmax.data.max(1)[0]
        # threshold
        # for i in range(len(pred_choice)):
        #     sample_counter += 1
        #     if pred_possi[i] < 0.95:  # prediction rejection threshold
        #         ignore_counter += 1
        #         continue
        #     if pred_choice[i] == target[i]:
        #         correct_counter += 1
        #     else:
        #         wrong_counter += 1
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    print("test_acc",test_acc)
    # print('with threshold: accuracy:', correct_counter / (sample_counter-ignore_counter))
    # print('with threshold: ignored sample ratio:', ignore_counter/sample_counter)
    return test_acc

#---- model test ----#
if __name__ == '__main__':
    # Load Dataset
    testDataLoader = DataLoader(MyDataLoader(opt,[0.8,1]), batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=True)
    print("num of sample: ",len(testDataLoader.dataset))
    
    eval(testDataLoader,opt.pretrained_parameters_path)

