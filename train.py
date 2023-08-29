from model.model_interface import *
import argparse
import sys

#---- parameter setting ----#
parser = argparse.ArgumentParser()
# model unrelated parameters (path etc.)
parser.add_argument("--log_dir", default="./log/", help="directory of log")
parser.add_argument("--data_dir", default="./data/60Ghz_radar/", help="directory of data")
parser.add_argument("--model_name", type=str, help="model name: mmgaitnet/srpnet/tcpcn/pointnet") # used for logging and model setting
parser.add_argument("--comment", type=str, default='no comment', help="comment on the experiment") 

# model related parameters
# training related hyper parameters
# HINT: there are 23 subjects in the data set however we just use 5 of them for demo of models in order to save training time.
#       adjust --num_classes to have more or less subjects.
parser.add_argument("--num_classes", type=int, default=5, help="number of classes for dataset")
# HINT: we use 60 frames here because the default setting(20 or 30) from the paper has a very low accuracy
parser.add_argument("--num_frames", type=int, default=60, help="number of frames for a sample")
parser.add_argument("--epoch", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--num_pts_per_frame", default=64, type=int, help="number of points per frame")
parser.add_argument("--save_acc",type=float,default=0.4, help="model are saved if its accuracy is higher than save_acc")
# classifier related hyper parameters
parser.add_argument("--lr", type=float, default=1e-3, help="classifier adam: learning rate")
parser.add_argument("--decay_rate", type=float, default=0, help="weight decay")
parser.add_argument("--label_smooth",type=float, default=0, help="label smoothing")
# tcpcn --lr 1e-4 --decay_rate 1e-3
#---- parameter check ----#
opt = parser.parse_args()
if opt.model_name is None:
    sys.exit("Model name is needed")

#---- model training ----#
if __name__ == '__main__':
    model = ModelInterface(opt)
    model.train()