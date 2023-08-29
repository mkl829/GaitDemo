from dataloader.dataloader_tools import *
from torch.utils.data import Dataset
import numpy as np


def load_data(raw_path, interval, num_frames, num_pts_per_frame, num_classes):
    """
    :param raw_path: directory of raw data
    :param interval: the interval of a sub-dataset
    :param num_frames: the number of frames of a sample
    :param num_pts_per_frame: the number of points should a frame have
    :param num_classes: the numnber of classes
    :return all_pcs: ([numOfFrames, numOfPoints, featureDim]), point cloud data
    :return idx_list: ([numOfSampleInSubset, 3] 3 elements=[SourceSubset, beginFrame, endFrame] (endFrame is NOT INCLUDED in a sample)
    :return all_label: ([numOfSampleInSubset, 1]), an integer index is used to imply a label
    """
    all_pcs, all_profiles = data_generator(raw_path, num_classes)
    all_pcs = pc_preprocess(all_pcs, interval, num_pts_per_frame, padding='zero', min_num_pts=5)
    len_data = [len(i) for i in all_pcs]
    labels = [c for c in range(len(all_profiles))]
    idx_list, all_label = dynamic_sample_building(len_data, num_frames, labels)

    print(len(all_pcs), len(idx_list), len(all_label))
    return all_pcs, idx_list, all_label       


class MyDataLoader(Dataset):
    def __init__(self, opt, interval):
        self.pcs, self.idx_list, self.labels = load_data(opt.data_dir, interval, opt.num_frames, opt.num_pts_per_frame, opt.num_classes)
        print(" sample num ", len(self.idx_list))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pc = self.pcs[self.idx_list[index, 0]][self.idx_list[index, 1]:self.idx_list[index, 2], :].copy()
        label = self.labels[index]
        return pc.astype(np.float32), label.astype(np.int32)