# QAConv
Interpretable and Generalizable Person Re-identification with Query-adaptive Convolution and Temporal Lifting

This PyTorch code is proposed in our paper [1]. 

# Requirements
Install Pytorch (>1.0) and its related dependencies, and then install sklearn.

# Usage
Download some public datasets (e.g. Market-1501, DukeMTMC-reID, CUHK03-NP, MSMT) on your own, extract them in some 
folder, and then run the followings.

## Training and test
python main.py --dataset market --testset duke [--data-dir ./data] [--exp-dir ./Exp]

## Test only
python main.py --dataset market --testset duke [--data-dir ./data] [--exp-dir ./Exp] --evaluate

# Performance

Performance (%) of direct cross-dataset evaluation without transfer learning or domain adaptation:

| Method | Training set | Test set | Rank-1 | mAP  |
| :----: | :----------: | :------: | :----: | :---: |
| QAConv |     Market   |   Duke   |  54.4  | 33.6 |
| QAConv+RR |     Market   |   Duke   |  61.8  | 52.4 |
| QAConv+RR+TLift |     Market   |   Duke   |  66.7  | 56.0 |
|  |
| QAConv |     MSMT   |   Duke   |  72.2  | 53.4 |
| QAConv+RR |     MSMT   |   Duke   |  78.1  | 72.4 |
| QAConv+RR+TLift |     MSMT   |   Duke   |  79.5  | 76.1 |
|  |
| QAConv |     Duke   |  Market | 62.1 | 31.0 |
| QAConv+RR |     Duke   |  Market | 68.2 | 51.2 |
| QAConv+RR+TLift |     Duke   |  Market | 74.4 | 56.6 |
|  |
| QAConv |     MSMT   |   Market   |  73.9  | 46.6 |
| QAConv+RR |     MSMT   |   Market   |  79.2  | 69.1 |
| QAConv+RR+TLift |     MSMT   |   Market   |  85.7  | 75.1 |
| |
| QAConv |     Market   |   MSMT   |  25.6  | 8.2 |
| QAConv+RR |     Market   |   MSMT   |  32.7  | 16.3 |
| |
| QAConv |     Duke   |   MSMT   |  31.8  | 10.0 |
| QAConv+RR |     Duke   |   MSMT   |  40.4  | 20.2 |
| |
| QAConv |     Market   |   CUHK03-NP   | 14.1 | 11.8 |
| QAConv+RR |     Market   |   CUHK03-NP   | 19.7 | 21.2 |
| |
| QAConv |     Duke   |   CUHK03-NP   | 10.9 | 9.2 |
| QAConv+RR |     Duke   |   CUHK03-NP   | 16.0 | 16.9 |
| |
| QAConv |     MSMT   |   CUHK03-NP   | 32.6 | 28.1 |
| QAConv+RR |     MSMT   |   CUHK03-NP   | 41.9 | 44.6 |

# Contacts

Shengcai Liao  
Inception Institute of Artificial Intelligence (IIAI)  
shengcai.liao@inceptioniai.org

# Citation
[1] Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-identification with Query-adaptive Convolution and Temporal Lifting." In the 16th European Conference on Computer Vision (ECCV), 23-28 August, 2020.

@inproceedings{Liao-ECCV2020-QAConv,  
  title={Interpretable and Generalizable Person Re-identification with Query-adaptive Convolution and Temporal Lifting},  
  author={Wen, Yandong and Zhang, Kaipeng and Li, Zhifeng and Qiao, Yu},  
  booktitle={European conference on computer vision (ECCV)},  
  year={2020}  
}
