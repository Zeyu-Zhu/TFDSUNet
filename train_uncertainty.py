import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from model.sffnet import USFFNet
from utils.evaluate import Evaluate
from utils.build_dataloader import get_dataloder


# Dataloader&&Evaluate
device = 'cuda:1'
batch_size = 256
stride = 1
window_size = 150
# path = 'datasets/10degree'
# train_list = ['567_Mixed1.csv', '567_Mixed2.csv', '571_Mixed4.csv', '571_Mixed5.csv', '571_Mixed6.csv', '571_Mixed7.csv']
# test_list = ['571_Mixed8.csv']
# path = 'datasets/0degree'
# train_list = ['589_Mixed1.csv', '589_Mixed2.csv', '590_Mixed4.csv', '590_Mixed5.csv', '590_Mixed6.csv', '590_Mixed7.csv']
# test_list = ['590_Mixed8.csv']
path = 'datasets/n10degree'
train_list = ['601_Mixed1.csv', '601_Mixed2.csv', '602_Mixed4.csv', '602_Mixed5.csv', '604_Mixed3.csv', '604_Mixed6.csv', '604_Mixed7.csv']
test_list = ['604_Mixed8.csv']
# path = 'datasets/n20degree'
# train_list = ['610_Mixed1.csv', '610_Mixed2.csv', '611_Mixed4.csv', '611_Mixed5.csv', '611_Mixed3.csv', '611_Mixed6.csv', '611_Mixed7.csv']
# test_list = ['611_Mixed8.csv']
train_loder, test_loader = get_dataloder(path, window_size, stride, train_list, test_list, batch_size, device)
# Trainning&&Model Config
test_ratio = 1
epoches = 100
weight_decay = 1e-4
learning_rate = 5e-4
loss_funcation = nn.MSELoss()
evaluater = Evaluate(path.split('/')[1], 'Ablation', test_ratio)
block_num = 5
feature_num = 3
spa_ks_list = [3, 5, 7, 7, 7]
fre_ks_list = [3, 5, 7, 7, 7]
fus_ks_list = [3, 3, 7, 7, 7]
mid_channel_list = [32, 16, 8, 4, 4]
model = USFFNet(block_num, feature_num, window_size, mid_channel_list, spa_ks_list, fre_ks_list, fus_ks_list).to(device)
# model.apply(weights_init)
optimizer = opt.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
evaluater.record_param_setting(window_size, stride, batch_size, learning_rate, weight_decay, model)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

# Trainning
lambda_coef = []
train_loss = []
vaild_loss = []
for epoch in range(epoches):
    model.train()
    epoch_loss = 0
    print('epoch: '+str(epoch))
    for x, y in tqdm(train_loder):
        # train
        gamma, nu, alpha, beta = model.forward(x)
        loss, nig_loss, nig_regularization = model.Uncertainty_Head.get_loss(y, gamma, nu, alpha, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.Uncertainty_Head.hyperparams_update(nig_regularization)
        # record
        _loss = loss_funcation(gamma, y)
        epoch_loss += _loss.item()
    epoch_loss /= train_loder.__len__()
    lambda_coef.append(model.Uncertainty_Head.lambda_coef.detach().cpu().numpy())
    train_loss.append(epoch_loss)
    print('trainning_loss = '+str(epoch_loss))
    if epoch%test_ratio == 0:
        model.eval()
        epoch_loss = 0
        for x, y in test_loader:
            gamma, nu, alpha, beta = model.forward(x)
            _loss = loss_funcation(gamma, y)
            epoch_loss += _loss.item()
        epoch_loss /= test_loader.__len__()
        vaild_loss.append(epoch_loss)
        print('testing_loss = '+str(epoch_loss))
    evaluater.visualize(train_loss, vaild_loss, model, None)
    evaluater.draw('lambda_coef', lambda_coef)