# GaitDemo: A small demo of gait recognition models based on mmWave radar point clouds
The repository contains a PyTorch re-implementation of the following gait recognition models: *mmGaitNet*, *SRPNet*, *TCPCN* and *PointNet*.

*PointNet* is a general classification model based on point clouds, while the other three models are specialized for gait recognition based on point clouds collected by millimeter wave radar.

For more information about these models, please refer to [**Related Papers**](#click_jump).

We use a much simplified data preprocess method and on these models.

We slightly modify these models so that they can be used on different number of frames and different number of points per frame.

## Dependencies
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
Download the gait dataset from <a href="https://github.com/mmGait/people-gait" target="_blank">here</a>.

Here we only focus on scenarios with single subject. We choose to use the data from "room1/1/fixed_route/60Ghz_radar" for demo.

To train a model:

    python train.py --model_name <model_name> --data_dir <dataset_folder_dir>

* `<model_name>` can be one of the following options: mmgaitnet, srpnet, tcpcn and pointnet.

* `<dataset_folder_dir>` is the directory where the dataset is located in.

* For more details of the parameter setting, please refer to `train.py`.

* If you want to have a small demo for all the models, just modify `<model_name>` to assign the model you want to use and leave other settings in default. (WARN: This may lead to suboptimal results.)

* Log files and network parameters will be saved to `log` folder in default.

To test a model:

    python pred.py --model_name  <model_name> --pretrained_parameters_path <pretrained_parameters_path>

* `<pretrained_parameters_path> is the path of pretrained_parameters ("xxx.pth").

## Implementation Details

* Implementation of models is located in `module` folder. Each model is placed in a separate file.

* Training code is located in `model` folder.
  * Basic tools of training (e.g., confusion matrix drawing, log setting) are placd in `model_prototype.py`.
  * Core code of training (e.g., forward propagation, loss calculation) is placed in `model_zoo.py`.
  * Framework code of training (e.g., data loading, basic model setting) is placed in `model_interface.py`.

* Data loading and preprocessing code is located in `dataloader` folder.
  * Framework code of a customized dataloader for data loading is placed in `dataloader.py`.
  * Data loading and preprocessing functions are placed in `dataloader_tools.py`.

* You use `train.py` to activate model training.
  
* You use `pred.py` to activate model testing.

## Related Papers
 <a id="click_jump"></a>
 <a href="https://ojs.aaai.org/index.php/AAAI/article/download/5430/5286">__mmGaitNet__</a>. Meng, Zhen, et al. "Gait recognition for co-existing multiple people using millimeter wave sensing." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 01. 2020.

 <a href="https://ieeexplore.ieee.org/abstract/document/9420713/">__SRPNet__</a>. Cheng, Yuwei, and Yimin Liu. "Person reidentification based on automotive radar point clouds." IEEE Transactions on Geoscience and Remote Sensing 60 (2021): 1-13.

 <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9440989">__TCPCN__</a>. Pegoraro, Jacopo, and Michele Rossi. "Real-time people tracking and identification from sparse mm-wave radar point-clouds." IEEE Access 9 (2021): 78504-78520.

 <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf">__PointNet__</a>. Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
