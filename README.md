# QAConv
Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting

This PyTorch code is proposed in our paper [1]. A Chinese blog is available in [再见，迁移学习？可解释和泛化的行人再辨识](https://mp.weixin.qq.com/s/ukZgCsGdig0jE6jmkpBbbA).

# Updates

* 4/1/2021: QAConv 2.0 [2]: include a new sampler called Graph Sampler (GS), and remove the class memory. This version is much more efficient in learning. See the updated [results](#Performance).
* 3/31/2021: QAConv 1.2: include some popular data augmentation methods, and change the ranking.py implementation to the original open-reid version, so that it is more consistent to most other implementations (e.g. open-reid, torch-reid, fast-reid).
* 2/7/2021: QAConv 1.1: an important update, which includes a pre-training function for a better initialization, so that the [results](#Performance) are now more stable.
* 11/26/2020: Include the IBN-Net as backbone, and the [RandPerson](https://github.com/VideoObjectSearch/RandPerson) dataset.

# Requirements

- Pytorch (>1.0)
- sklearn
- scipy

# Usage
Download some public datasets (e.g. Market-1501, CUHK03-NP, MSMT) on your own, extract them in some 
folder, and then run the followings.

## Training and test
`python main.py --dataset market --testset cuhk03_np_detected[,msmt] [--data-dir ./data] [--exp-dir ./Exp]`

For more options, run "python main.py --help". For example, if you want to use the ResNet-152 as backbone, specify "-a resnet152". If you want to train on the whole dataset (as done in our paper for the MSMT17), specify "--combine_all".

With the GS sampler and pairwise matching loss, run the following:

``python main_gs.py --dataset market --testset cuhk03_np_detected[,msmt] [--data-dir ./data] [--exp-dir ./Exp]``

## Test only
`python main.py --dataset market --testset duke[,market,msmt] [--data-dir ./data] [--exp-dir ./Exp] --evaluate`

# Performance

Updated performance (%) of QAConv under direct cross-dataset evaluation without transfer learning or domain adaptation:

<table align="center">
  <tr align="center">
    <td rowspan="2">Training Data</td>
    <td rowspan="2">Version</td>
    <td rowspan="2">Training Time (h)</td>
    <td colspan="2">CUHK03-NP</td>
    <td colspan="2">Market-1501</td>
    <td colspan="2">MSMT17</td>
  </tr>
  <tr align="center">
    <td>Rank-1</td>
    <td>mAP</td>
    <td>Rank-1</td>
    <td>mAP</td>
    <td>Rank-1</td>
    <td>mAP</td>
  </tr>
  <tr align="center">
    <td rowspan="4">Market</td>
    <td>QAConv 1.0</td>
    <td>1.33</td>
    <td>9.9</td>
    <td>8.6</td>
    <td>-</td>
    <td>-</td>
    <td>22.6</td>
    <td>7.0</td>
  </tr>
  <tr align="center">
    <td>QAConv 1.1</td>
    <td>1.02</td>
    <td>12.4</td>
    <td>11.3</td>
    <td>-</td>
    <td>-</td>
    <td>35.6</td>
    <td>12.2</td>
  </tr>
  <tr align="center">
    <td>QAConv 1.2</td>
    <td>1.07</td>
    <td>13.3</td>
    <td>14.2</td>
    <td>-</td>
    <td>-</td>
    <td>40.9</td>
    <td>14.7</td>
  </tr>
  <tr align="center">
    <td>QAConv 2.0</td>
    <td><b>0.68</b></td>
    <td><b>16.4</b></td>
    <td><b>15.7</b></td>
    <td>-</td>
    <td>-</td>
    <td><b>41.2</b></td>
    <td><b>15.0</b></td>
  </tr>
  <tr align="center">
    <td rowspan="2">MSMT</td>
    <td>QAConv 1.2</td>
    <td>2.37</td>
    <td>15.6</td>
    <td>16.2</td>
    <td>72.9</td>
    <td>44.2</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>QAConv 2.0</td>
    <td><b>0.96</b></td>
    <td><b>20.0</b></td>
    <td><b>19.2</b></td>
    <td><b>75.1</b></td>
    <td><b>46.7</b></td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td rowspan="4">MSMT (all)</td>
    <td>QAConv 1.0</td>
    <td>26.90</td>
    <td>25.3</td>
    <td>22.6</td>
    <td>72.6</td>
    <td>43.1</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>QAConv 1.1</td>
    <td>18.16</td>
    <td>27.1</td>
    <td>25.0</td>
    <td>76.0</td>
    <td>47.9</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>QAConv 1.2</td>
    <td>17.85</td>
    <td>25.1</td>
    <td>24.8</td>
    <td>79.5</td>
    <td>52.3</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>QAConv 2.0</td>
    <td><b>3.88</b></td>
    <td><b>27.2</b></td>
    <td><b>27.1</b></td>
    <td><b>80.6</b></td>
    <td><b>55.6</b></td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td rowspan="3">RandPerson</td>
    <td>QAConv 1.1</td>
    <td>12.05</td>
    <td>12.9</td>
    <td>10.8</td>
    <td>68.0</td>
    <td>36.8</td>
    <td>36.6</td>
    <td>12.1</td>
  </tr>
  <tr align="center">
    <td>QAConv 1.2</td>
    <td>12.22</td>
    <td>12.6</td>
    <td>12.1</td>
    <td>73.2</td>
    <td>42.1</td>
    <td>41.8</td>
    <td>13.8</td>
  </tr>
  <tr align="center">
    <td>QAConv 2.0</td>
    <td><b>1.84</b></td>
    <td><b>14.8</b></td>
    <td><b>13.4</b></td>
    <td><b>74.0</b></td>
    <td><b>43.8</b></td>
    <td><b>42.4</b></td>
    <td><b>14.4</b></td>
  </tr>
</table>

**Version Difference:**

| Version    | Backbone  | IBN Type | Pre-trials | Loss              | Sampler | Data Augmentation |
| ---------- | --------- | -------- | ---------- | ----------------- | ------- | ----------------- |
| QAConv 1.0 | ResNet-50 | None     | x          | Class Memory      | Random  | Old               |
| QAConv 1.1 | ResNet-50 | b        | √          | Class Memory      | Random  | Old               |
| QAConv 1.2 | ResNet-50 | b        | √          | Class Memory      | Random  | New               |
| QAConv 2.0 | ResNet-50 | b        | x          | Pairwise Matching | GS      | New               |

**Notes:** 

* Except QAConv 1.0, the other versions additionally include three IN layers as in IBN-Net-b.
* QAConv 1.1 and 1.2 additionally include a pre-training function with 10 trials to stable the results.
* QAConv 1.2 and 2.0 additionally apply some popular data augmentation methods.
* QAConv 2.0 applies the GS sampler and the pairwise matching loss.
* QAConv 1.0 results are obtained by neck=128, batch_size=32, lr=0.01, epochs=60, and step_size=40, trained with two V100.
* QAConv 1.1 and 1.2 results are obtained by neck=64, batch_size=8, lr=0.005, epochs=15, and step_size=10 (except for RandPerson epochs=4 and step_size=2), trained on one single V100.
* QAConv 2.0 results are obtained by neck=64, batch_size=64, K=4, lr=0.001, epochs=15, and step_size=10 (except for RandPerson epochs=4 and step_size=2), trained on one single V100.

# Pre-trained Models for QAConv 1.0 on ECCV 2020

- [QAConv_ResNet50_MSMT](https://1drv.ms/u/s!Ak6Huh3i3-MzdRN84Kd6Xrn5FXg?e=cJmCui)
- [QAConv_ResNet152_MSMT](https://1drv.ms/u/s!Ak6Huh3i3-MzdhATpabUgh5f2aY?e=RD8tRV)

The above pre-trained models can also be downloaded from [Baidu](https://pan.baidu.com/s/1fe3PliWl-mmYQAu5nhSJ8A) (access code: 52cv), thanks to [52CV](https://mp.weixin.qq.com/s/HHINgdVchZuSeTUPV8E4GQ).

# Contacts

Shengcai Liao  
Inception Institute of Artificial Intelligence (IIAI)  
shengcai.liao@inceptioniai.org

# Citation
[1] Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting." In the 16th European Conference on Computer Vision (ECCV), 23-28 August, 2020.

[2] Shengcai Liao and Ling Shao, "Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification." In arXiv preprint, arXiv:2104.01546, 2021.

```
@inproceedings{Liao-ECCV2020-QAConv,  
  title={{Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting}},  
  author={Shengcai Liao and Ling Shao},  
  booktitle={European Conference on Computer Vision (ECCV)},  
  year={2020}  
}

@article{Liao-arXiv2021-GS,
  author    = {Shengcai Liao and Ling Shao},
  title     = {{Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification}},
  journal   = {CoRR},
  volume    = {abs/2104.01546},
  year      = {April 4, 2021},
  url       = {http://arxiv.org/abs/2104.01546},
  archivePrefix = {arXiv},
  eprint    = {2104.01546}
}
```