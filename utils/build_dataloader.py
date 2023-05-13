import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_data(path, file_list, window_size, stride, device):
    window_data_x = []
    window_data_y = []
    SOC = np.array([0])
    for file_name in tqdm(file_list):
        data = pd.read_csv(os.path.join(path, file_name), skiprows=30)
        data.columns = ['Time Stamp', 'Step', 'Status', 'Prog Time', 'Step Time', 'Cycle', 'Cycle Level', 'Procedure',
                        'Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu', 'Cnt', 'Empty']
        data = data[(data["Status"] == "TABLE") | (data["Status"] == "DCH")]
        # Normalize SOC
        max_discharge = abs(min(data["Capacity"]))
        data["SoC Capacity"] = max_discharge + data["Capacity"] 
        data["SoC Percentage"] = data["SoC Capacity"] / max(data["SoC Capacity"]) 
        y = data[["SoC Percentage"]].to_numpy()
        # Normalize Voltage, Current, Temperature 
        x = data[["Voltage", "Current", "Temperature"]].to_numpy()
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
        # Generate window trainning data
        for start in range(0, x.shape[0] - window_size, stride):
            end = start + window_size
            window_x = x[start:end, ...]
            window_y = y[end-1]
            window_data_x.append(window_x)
            window_data_y.append(window_y)
    return window_data_x, window_data_y
        
def get_dataloder(path, window_size, stride, train_list, test_list, batch_size, device, test=False):
    print('loading data...')
    
    if test == False:
        train_x, train_y = load_data(path, train_list, window_size, stride, device)
        train_x, train_y = torch.Tensor(np.array(train_x)).to(device).float().transpose(1, 2), torch.Tensor(np.array(train_y)).to(device).float()
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        train_loader = None
    test_x, test_y = load_data(path, test_list, window_size, stride, device)
    test_x, test_y = torch.Tensor(np.array(test_x)).to(device).float().transpose(1, 2), torch.Tensor(np.array(test_y)).to(device).float()
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader
    
        
if __name__ == '__main__':
    path = '/home/cxy/pen-sharpening/SOC_Predication/datasets/0degree'
    train_names = [
        '589_Mixed1.csv',
        '589_Mixed2.csv',
        '590_Mixed4.csv',
        '590_Mixed5.csv',
        '590_Mixed6.csv',
        '590_Mixed7.csv',
        '590_Mixed8.csv']
    test_names = ['589_LA92.csv']
    device = 'cuda:0'
    train_loader, test_loader = get_dataloder(path, 100, 10, train_names, test_names, 32, device)
    # for x, y in train_loader:
    #     print(x.shape, y.shape)
   