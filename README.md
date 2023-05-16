# TFDSUNet: Time-Frequency Dual-Stream Uncertainty Network for Battery SOH/SOC Prediction
[`Wenzhe Xiao`](https://gr.xjtu.edu.cn/en/web/caoxiangyong/home), and [`Zeyu Zhu`](https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&gmla=ABEO0Ypgw7n86h8mMjkhHVfmhMuPPgnO7C4NT-RWQ_lB1xSqtIWcyPqrPOsxI7ffxZ-amtiKK7KVFRnx_ZOPxFYH2-iKKLY&user=X3CisOwAAAAJ)


## Our Insight
<p align="center">
  <img src="figure/insight.png" />
</p>
The sequence of current, voltage and temperature presents a strong periodicity during battery discharge. Existing data-driven methods often only explore feature extraction in the time domain. However, the invariance and variability in the frequency domain are better data structures for data extraction.And the local feature extraction in the frequency domain corresponds to the global extraction in the time domain. Our method can also be regarded as a global information supplement to the local feature extraction in the time domain.


## Set up a virtual conda environment
Setup a virtual conda environment using the provided ``requirements.txt``.
```
conda create --name TFDSUNet --file requirements.txt
conda activate TFDSUNet
```

## Download Datasets 
Datasets can be found in [`CALCE`](https://calce.umd.edu/battery-data#Citations) and [`LG`](https://data.mendeley.com/datasets/cp3473x7xv/3). You can download and put it into `datastets`. You can also apply our model on other datasets, but then you have to modify `utils/build_dataloader` to make it suitable to your data struct.


## Train and Test TFDSUNet
<p align="center">
  <img src="figure/main.png" />
</p>

The implementation of TFDSUNet is in `model`, including uncertainty head, frequency domain flow, spatial domain flow and dual-stream flow.After setting up a virtual environment and download the datasets, if you want to train your own model, please run following order in your terminal.
```
python train_uncertainty.py
```
And if you want to test it, please run following order in your terminal.
```
python test_uncertainty.py
```
It's worthing noticing that you may need to modify `path` in `train_uncertainty.py` because you may change name of files.
## Other Codes
Data preprocessing, dataloader and metrics(MSE and RMSR) are implemented in `utils`.

## Result on LG Datasets
Pre-trained models on 0 degree and 10 degree datasets are saved in `result/0degree` and `result/10degree`, respectively.