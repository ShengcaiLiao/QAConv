# QAConv
Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting

This PyTorch code is proposed in our paper [1]. A Chinese blog is available in [再见，迁移学习？可解释和泛化的行人再辨识](https://mp.weixin.qq.com/s/ukZgCsGdig0jE6jmkpBbbA).

# Updates

* 3/31/2021: Include some popular data augmentation methods, and change the ranking.py implementation to the original open-reid version, so that it is more consistent to most other implementations (e.g. open-reid, torch-reid, fast-reid).
* 2/7/2021: An important update: include a pre-training function for a better initialization, so that the [results](#Performance) are now more stable.
* 11/26/2020: Include the IBN-Net as backbone, and the [RandPerson](https://github.com/VideoObjectSearch/RandPerson) dataset.

# Requirements

- Pytorch (>1.0)
- sklearn
- scipy

# Usage
Download some public datasets (e.g. Market-1501, DukeMTMC-reID, CUHK03-NP, MSMT) on your own, extract them in some 
folder, and then run the followings.

## Training and test
python main.py --dataset market --testset duke[,market,msmt] [--data-dir ./data] [--exp-dir ./Exp]

For more options, run "python main.py --help". For example, if you want to use the ResNet-152 as backbone, specify "-a resnet152". If you want to train on the whole dataset (as done in our paper for the MSMT17), specify "--combine_all".

## Test only
python main.py --dataset market --testset duke[,market,msmt] [--data-dir ./data] [--exp-dir ./Exp] --evaluate

# Performance

* Updated performance (%) of QAConv under direct cross-dataset evaluation without transfer learning or domain adaptation:
<table>
  <tr>
    <td rowspan="3">Backbone</td>
    <td rowspan="3">Training set</td>
    <td colspan="8" align="center">Test set</td>
  </tr>
  <tr>
    <td colspan="2" align="center">Market</td>
    <td colspan="2" align="center">Duke</td>
    <td colspan="2" align="center">CUHK</td>
    <td colspan="2" align="center">MSMT</td>
  </tr>
  <tr>
    <td>Rank-1</td>
    <td>mAP</td>
    <td>Rank-1</td>
    <td>mAP</td>
    <td>Rank-1</td>
    <td>mAP</td>
    <td>Rank-1</td>
    <td>mAP</td>
  </tr>
  <tr>
    <td rowspan="3">ResNet-50</td>
    <td>Market</td>
    <td>-</td>
    <td>-</td>
    <td>49.5</td>
    <td>29.7</td>
    <td>10.6</td>
    <td>9.3</td>
    <td>26.4</td>
    <td>8.3</td>
  </tr>
  <tr>
    <td>MSMT (all)</td>
    <td>73.8</td>
    <td>44.1</td>
    <td>69.7</td>
    <td>51.8</td>
    <td>24.6</td>
    <td>22.8</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>RandPerson</td>
    <td>65.6</td>
    <td>34.8</td>
    <td>59.4</td>
    <td>36.1</td>
    <td>14.3</td>
    <td>11.0</td>
    <td>34.3</td>
    <td>10.7</td>
  </tr>
  <tr>
    <td rowspan="3">IBN-Net-b (ResNet-50)</td>
    <td>Market</td>
    <td>-</td>
    <td>-</td>
    <td>54.0</td>
    <td>35.0</td>
    <td>12.4</td>
    <td>11.3</td>
    <td>35.6</td>
    <td>12.2</td>
  </tr>
  <tr>
    <td>MSMT (all)</td>
    <td>76.0</td>
    <td>47.9</td>
    <td>71.6</td>
    <td>53.6</td>
    <td>27.1</td>
    <td>25.0</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>RandPerson</td>
    <td>68.0</td>
    <td>36.8</td>
    <td>61.7</td>
    <td>38.9</td>
    <td>12.9</td>
    <td>10.8</td>
    <td>36.6</td>
    <td>12.1</td>
  </tr>
</table>
    
Note: results are obtained by neck=64, batch_size=8, lr=0.005, epochs=15, and step_size=10 (except for RandPerson epochs=4 and step_size=2), trained on one single GPU. By this setting the traininig and testing time and memory is much reduced.

* Performance (%) of QAConv in the ECCV paper, with ResNet-152 under direct cross-dataset evaluation:

| Method | Training set | Test set | Rank-1 | mAP  |
| :----: | :----------: | :------: | :----: | :---: |
| QAConv |     Market   |   Duke   |  54.4  | 33.6 |
| QAConv + RR + TLift |     Market   |   Duke   |  70.0  | 61.2 |
|  |
| QAConv |     MSMT   |   Duke   |  72.2  | 53.4 |
| QAConv + RR + TLift |     MSMT   |   Duke   |  82.2  | 78.4 |
|  |
| QAConv |     Duke   |  Market | 62.8 | 31.6 |
| QAConv + RR + TLift |     Duke   |  Market | 78.7 | 58.2 |
|  |
| QAConv |     MSMT   |   Market   |  73.9  | 46.6 |
| QAConv + RR + TLift |     MSMT   |   Market   |  88.4  | 76.0 |
| |
| QAConv |     Market   |   MSMT   |  25.6  | 8.2 |
| QAConv |     Duke   |   MSMT   |  32.7  | 10.4 |
| |
| QAConv |     Market   |   CUHK03-NP   | 14.1 | 11.8 |
| QAConv |     Duke   |   CUHK03-NP   | 11.0 | 9.4 |
| QAConv |     MSMT   |   CUHK03-NP   | 32.6 | 28.1 |

# Pre-trained Models

- [QAConv_ResNet50_MSMT](https://1drv.ms/u/s!Ak6Huh3i3-MzdRN84Kd6Xrn5FXg?e=cJmCui)
- [QAConv_ResNet152_MSMT](https://1drv.ms/u/s!Ak6Huh3i3-MzdhATpabUgh5f2aY?e=RD8tRV)

The above pre-trained models can also be downloaded from [Baidu](https://pan.baidu.com/s/1fe3PliWl-mmYQAu5nhSJ8A) (access code: 52cv), thanks to [52CV](https://mp.weixin.qq.com/s/HHINgdVchZuSeTUPV8E4GQ).

# Contacts

Shengcai Liao  
Inception Institute of Artificial Intelligence (IIAI)  
shengcai.liao@inceptioniai.org

# Citation
[1] Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting." In the 16th European Conference on Computer Vision (ECCV), 23-28 August, 2020.

@inproceedings{Liao-ECCV2020-QAConv,  
  title={{Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting}},  
  author={Shengcai Liao and Ling Shao},  
  booktitle={European Conference on Computer Vision (ECCV)},  
  year={2020}  
}
