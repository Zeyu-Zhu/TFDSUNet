import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.build_dataloader import get_dataloder
from utils.metrics import *


def Test(test_dataloder, model, path, data_type):
    # Testing
    MAE = []
    MSE = []
    y_true = np.array([1])
    y_hat =  np.array([1])
    upper_bound_list = np.array([1])
    lower_bound_list = np.array([1])
    for x, y in tqdm(test_dataloder):
        gamma, nu, alpha, beta = model.forward(x)
        MAE.append(getMAE(gamma, y))
        MSE.append(getMSE(gamma, y))
        alpha = alpha.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy()
        beta = beta.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy()
        gamma = gamma.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy()
        var = np.sqrt(beta / (alpha - 1))
        upper_bound = gamma + 1.96*var
        lower_bound = gamma - 1.96*var
        y = y.squeeze(dim=-1).squeeze(dim=-1).detach().cpu().numpy()
        y_hat = np.concatenate((y_hat, gamma))
        y_true = np.concatenate((y_true, y))
        upper_bound_list = np.concatenate((upper_bound_list, upper_bound))
        lower_bound_list = np.concatenate((lower_bound_list, lower_bound))
    MAE = np.array(MAE).mean()
    RMSE = np.sqrt(np.array(MSE).mean())
    y_hat = np.array(y_hat)
    record = pd.DataFrame(np.transpose(np.array([y_hat, y_true, lower_bound_list, upper_bound_list]), [1, 0]))
    record.columns = ['y_hat', 'y_true', 'lower_bound_list', 'upper_bound_list']
    record.to_excel(path+'/'+'result.xlsx')
    plt.grid(color='#7d7f7c', linestyle='-.')
    plt.plot(np.arange(len(y_hat)), y_hat, 'b', linewidth=0.1)
    plt.plot(np.arange(len(y_hat)), y_true, 'r', linewidth=0.5)
    plt.fill_between(np.arange(len(y_hat)), lower_bound_list, upper_bound_list, facecolor='blue', alpha=0.5)
    plt.title(data_type)
    plt.xlabel('time step')
    plt.ylabel('SOC')
    plt.ylim(0, 1)
    plt.savefig(path+'/'+data_type+'.jpg', dpi=300)
    plt.clf()
    return MAE, RMSE

# Dataloaer
device = 'cuda:1'
batch_size = 64
stride = 1
window_size = 150
# path = 'datasets/10degree'
# train_list = ['567_Mixed1.csv', '567_Mixed2.csv', '571_Mixed4.csv', '571_Mixed5.csv', '571_Mixed6.csv', '571_Mixed7.csv']
# test_list = ['571_Mixed8.csv']
path = 'datasets/0degree'
train_list = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv']
test_list = ['590_Mixed8.csv']
# path = 'datasets/n10degree'
# train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed3.csv', '604_Mixed6.csv', '604_Mixed7.csv']
# test_list = ['604_Mixed8.csv']
# path = 'datasets/n20degree'
# train_list = ['610_Mixed1.csv', '610_Mixed2.csv', '611_Mixed4.csv', '611_Mixed5.csv', '611_Mixed3.csv', '611_Mixed6.csv', '611_Mixed7.csv']
# test_list = ['611_Mixed8.csv']
# exp_path = 'result/BBM_0degree/USFFNet/exp0'
exp_path = 'result/'+path.split('/')[-1]+'/Ablation/exp2'
model = torch.load(os.path.join(exp_path, 'model.pkl')).to(device)

# path = 'datasets/0degree'
# train_list = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv']
# test_list = ['590_Mixed8.csv']
# path = 'datasets/n10degree'
# train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed3.csv', '604_Mixed6.csv', '604_Mixed7.csv']
# test_list = ['604_Mixed8.csv']
result = open(os.path.join(exp_path, 'result.txt'), mode='w')
train_loder, test_loder = get_dataloder(path, window_size, stride, train_list, test_list, batch_size, device, True)
MAE, RMSE = Test(test_loder, model, exp_path, 'Result')
result.write('Result: MAE='+str(MAE)+', RMSE='+str(RMSE)+'\n')
result.close()