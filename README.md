# GaitDemo: A demo of gait recognition models based on mmWave radar point clouds
The repository contains a PyTorch re-implementation of the following gait recognition models: *mmGaitNet*, *SRPNet*, *TCPCN* and *PointNet*.

*PointNet* is a general classification model based on point clouds, we have modified it to fit the data. The other three models are specialized for gait recognition based on point clouds collected by millimeter wave radar.

For more information about these models, please refer to [**Related Papers**](#click_jump).

Pay attention that we don't follow the original data preprocess method on the papers. Instead, we use a much simplified data preprocess method on these models. Besides, we slightly modify these models so that they can be used on different number of frames and different number of points per frame.

## Installation
The code has been tested with Python 3.9.17, CUDA 11.7 and PyTorch 1.13.1 on Windows 11.

Following is the suggested way to install the dependencies: 
```bash
# Create a new conda environment
conda create -n GaitDemo python==3.9.17
conda activate GaitDemo

# Install pytorch (please refer to the command in the official website)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other necessary packages
pip install matplotlib numpy pandas scikit-learn tensorboardX tqdm

# Or just use the following command to install all the packages
pip install -r requirements.txt
```

## Usage
We use an open gait dataset from <a href="https://github.com/mmGait/people-gait" target="_blank">here</a>. It is published by the authors of *mmGaitNet*.

We only focus on scenarios with **single** subject and we only use **single** radar, which is somehow different from what is done in *mmGaitNet*, which tries to combine data from two radars.  We choose to use the data from "room1/1/fixed_route/60Ghz_radar" of the above dataset for demo, just as the same as "data/60Ghz_radar" in our repository.

To train a model:

    python train.py --model_name <model_name> --data_dir <dataset_folder_dir>

* `<model_name>` can be one of the following options: `mmgaitnet`, `srpnet`, `tcpcn` and `pointnet`.

* `<dataset_folder_dir>` is the directory where the dataset is located in. You can omit this parameter if you directly download this repository and don't move the dataset to other directories.

* For more details of the parameter setting, please refer to `train.py`.

* If you want to have a small demo for all the models, just modify `<model_name>` to assign the model you want to use and leave other settings in default. (WARN: This may lead to suboptimal results.)

* Log files and network parameters will be saved to `log` folder in default.

To test a model:

    python pred.py --model_name  <model_name> --pretrained_parameters_path <pretrained_parameters_path>

* `<pretrained_parameters_path>` is the path of pretrained_parameters like `"xxx.pth"`.  Some pretrained parameters are placed in `pretrained_models` folder.

## Implementation Details

* Implementation of models is located in `module` folder. Each model is placed in a separate file.

* Training code is located in `model` folder.
  * Basic tools of training (e.g., confusion matrix drawing, log setting) are placd in `model_prototype.py`.
  * Core code of training (e.g., for-/backward propagation, loss calculation) is placed in `model_zoo.py`.
  * Framework code of training (e.g., data loading, basic model setting) is placed in `model_interface.py`.

* Data loading and preprocessing code is located in `dataloader` folder.
  * Framework code of a customized dataloader for data loading is placed in `dataloader.py`.
  * Data loading and preprocessing functions are placed in `dataloader_tools.py`.

* `train.py` is used to activate model training.
  
* `pred.py` is used to activate model testing.

## Results
We use the data from 5 subjects for the experiment, their IDs are 006, 011, 042, 045 and 046.

We split the dataset into training and validation dataset with the ratio 80 : 20. No test dataset is allocated currently. We just report the best validation accuracy the models achieved, therefore it is just a rough (possibly unfair!) comparison of the models. The results are as follow.

|Model|Average Accuracy(%)|
|--|--|
|mmGaitNet|45.9|
|SRPNet|37.4|
|TCPCN|49.7|
|PointNet|50.6|

Pay attention that here we set --lr 1e-4 for *SRPNet*. We did not adopt all the suggested settings from the papers since sometimes they are not suitable for our situation. We will fine-tune the parameters and redo the experiments to achieve a more rigorous results in the future.

## Related Papers
 <a id="click_jump"></a>
 <a href="https://ojs.aaai.org/index.php/AAAI/article/download/5430/5286">__mmGaitNet__</a>. Meng, Zhen, et al. "Gait recognition for co-existing multiple people using millimeter wave sensing." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 01. 2020.

 <a href="https://ieeexplore.ieee.org/abstract/document/9420713/">__SRPNet__</a>. Cheng, Yuwei, and Yimin Liu. "Person reidentification based on automotive radar point clouds." IEEE Transactions on Geoscience and Remote Sensing 60 (2021): 1-13.

 <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9440989">__TCPCN__</a>. Pegoraro, Jacopo, and Michele Rossi. "Real-time people tracking and identification from sparse mm-wave radar point-clouds." IEEE Access 9 (2021): 78504-78520.

 <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf">__PointNet__</a>. Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
