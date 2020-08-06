# QAConv
Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting

This PyTorch code is proposed in our paper [1]. A Chinese blog is available in [再见，迁移学习？可解释和泛化的行人再辨识](https://mp.weixin.qq.com/s/ukZgCsGdig0jE6jmkpBbbA).

# Requirements
- Pytorch (>1.0)
- sklearn

# Usage
Download some public datasets (e.g. Market-1501, DukeMTMC-reID, CUHK03-NP, MSMT) on your own, extract them in some 
folder, and then run the followings.

## Training and test
python main.py --dataset market --testset duke[,market,msmt] [--data-dir ./data] [--exp-dir ./Exp]

For more options, run "python main.py --help". For example, if you want to use the ResNet-152 as backbone, specify "-a resnet152". If you want to train on the whole dataset (as done in our paper for the MSMT17), specify "--combine_all".

## Test only
python main.py --dataset market --testset duke[,market,msmt] [--data-dir ./data] [--exp-dir ./Exp] --evaluate

# Performance

Performance (%) of QAConv with ResNet-152 under direct cross-dataset evaluation without transfer learning or domain adaptation:

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
