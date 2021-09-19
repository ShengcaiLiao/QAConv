# Updates

* 9/16/2021: QAConv 2.1: simplify graph sampling, implement the Einstein summation for QAConv, use the batch hard triplet loss, design an adaptive epoch and learning rate scheduling method, and apply the automatic mixed precision training.
* 4/1/2021: QAConv 2.0 [2]: include a new sampler called Graph Sampler (GS), and remove the class memory. This version is much more efficient in learning. See the updated [results](#Performance).
* 3/31/2021: QAConv 1.2: include some popular data augmentation methods, and change the ranking.py implementation to the original open-reid version, so that it is more consistent to most other implementations (e.g. open-reid, torch-reid, fast-reid).
* 2/7/2021: QAConv 1.1: an important update, which includes a pre-training function for a better initialization, so that the [results](#Performance) are now more stable.
* 11/26/2020: Include the IBN-Net as backbone, and the [RandPerson](https://github.com/VideoObjectSearch/RandPerson) dataset.

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
