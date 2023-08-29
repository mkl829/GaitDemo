import pandas as pd
import numpy as np
import os

def data_translator(csv_path):
    """
    :param csv_path: path of raw data ("ddd.csv")
    :return pts: the point cloud data from csv_path
    """
    raw_data = pd.read_csv(csv_path)
    raw_data['pts'] = raw_data[['X', 'Y', 'Z', 'Doppler', 'Unnamed: 6']].values.tolist()
    pts = raw_data.groupby(by=['Frame #'])['pts'].apply(lambda x: list(x)).reset_index() #
    print("file_name: {} | num_raw_frames: {}".format(csv_path[-7:], len(pts)))
    return pts

def data_generator(raw_path, num_classes):
    """
    :param raw_path: directory of raw dataset
    :param num_classes: number of classes
    :return all_pcs: point cloud data
    :return all_profiles: labels
    """
    file_ls = [f for f in os.listdir(raw_path)]
    file_ls = file_ls[:num_classes] # to give a small demo of the models
    all_pcs, all_profiles = [], []
    for file_name in file_ls:
        data_csv_path = os.path.join(raw_path, file_name)
        output = data_translator(data_csv_path)
        all_pcs.append(output)
        all_profiles.append(file_name[:3])
    return all_pcs, all_profiles

def pc_preprocess(all_pcs_data, interval, num_pts_per_frame, padding='zero', min_num_pts=5):
    """
    :param all_pcs_data: point cloud data
    :param interval: allocate which part of raw data to use
    :param num_pts_per_frame: number of points per frame
    :param padding: "zero" or "valid", "zero" means padding with all-zero-value points, "valid" means padding with existed points
    :param min_num_pts: if a raw point cloud has less than min_num_pts points, then it will be discarded.
    :return preprocessed_data: preprocessed point cloud data
    """    
    preprocessed_data = []

    for sub_data in all_pcs_data:
        preprocessed_pc = []

        for ind in range(int(len(sub_data)*interval[0]), int(len(sub_data)*interval[1])):
            frame = sub_data.iloc[ind]['pts']
            frame = np.array(frame)
            
            if len(frame) < min_num_pts:
                continue
            centroid = np.mean(frame[:, :3], axis=0)
            frame[:,:3] -= centroid
            doppler_absmax = np.max(np.abs(frame[:, 3]))
            snr_absmax = np.max(np.abs(frame[:, 4]))
            if doppler_absmax > 0:
                frame[:, 3] /= doppler_absmax
            if snr_absmax > 0:
                frame[:, 4] /= snr_absmax
          
            if len(frame) > num_pts_per_frame:
                new_frame = frame[np.random.choice(len(frame), num_pts_per_frame, replace=False)]
            elif len(frame) <= num_pts_per_frame:
                if padding == "zero":
                    new_frame = np.zeros((num_pts_per_frame, 5))
                    new_frame[:len(frame)] = frame
                elif padding == "valid":
                    new_frame[len(frame):] = frame[np.random.choice(len(frame), num_pts_per_frame-len(frame), replace=True)]
                else:
                    print("Invalid padding setting in function **preprocess_pcs**.")
            preprocessed_pc.append(new_frame)
            
        preprocessed_pc = np.array(preprocessed_pc)
        preprocessed_data.append(preprocessed_pc)
        
    return preprocessed_data

def dynamic_sample_building(len_data, static_sample_length, labels):
    """
    :param len_data: number of frames in each sub-dataset 
    :param static_sample_length: the length of sample
    :param labels: label of each sub-dataset
    :return idx_list([num_of_sample][3]): list of the interval of each sample [serial number of the subset | start frame | end frame + 1]
    :return all_labels([num_of_sample][1]): list of the label of each sample
    """
    # interval between adjacent windows is 1 frame
    idx_list = []
    all_labels = []
    for i in range(len(len_data)):
        for j in range(len_data[i] - static_sample_length + 1):
            idx_list.append([i, j, j + static_sample_length])  # sample-id, first frame, frame after last frame
            all_labels.append([labels[i]])
    idx_list = np.array(idx_list)
    all_labels = np.array(all_labels)
    return idx_list, all_labels
