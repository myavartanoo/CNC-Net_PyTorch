# CNC-Net_PyTorch

This repository contains the official code to reproduce the results from the paper:

**𝐂𝐍𝐂-𝐍𝐞𝐭: 𝐒𝐞𝐥𝐟-𝐒𝐮𝐩𝐞𝐫𝐯𝐢𝐬𝐞𝐝 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐟𝐨𝐫 𝐂𝐍𝐂 𝐌𝐚𝐜𝐡𝐢𝐧𝐢𝐧𝐠 𝐎𝐩𝐞𝐫𝐚𝐭𝐢𝐨𝐧𝐬**

\[[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yavartanoo_CNC-Net_Self-Supervised_Learning_for_CNC_Machining_Operations_CVPR_2024_paper.html)\]\[[arXiv](https://arxiv.org/abs/2312.09925)\] \[[YouTube](https://www.youtube.com/watch?v=0wg5aV-q7XU&t=1s)\]

![architecture](https://github.com/myavartanoo/CNC-Net_PyTorch/blob/master/source/Framework.png)

## Installation
Clone this repository into any place you want.
```
git clone https://github.com/myavartanoo/CNC-Net_PyTorch.git
cd CNC-Net_PyTorch
```

### Citation
If you find our code or paper useful, please consider citing:
```
@inproceedings{CNCNet,
  title={CNC-Net: Self-Supervised Learning for CNC Machining Operations},
  author={Mohsen Yavatanoo and Sangmin Hong and Reyhaneh Neshatavar and Kyoung Mu Lee},
  booktitle={The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024}
}
```

### Dependencies
Install the dependencies:
* Python 3.8.18
* PyTorch 1.12.1
* numpy
* open3d
* pytorch3d
* pysdf

## Quick Start
To train the model for a given object individually run the below command.
```python train.py --input_object {objec_name} --gpu {device_index} --experiment {experiment_name}
```
For example:
```python train.py --input_object '0.off' --gpu 0 --experiment exp_0
```
