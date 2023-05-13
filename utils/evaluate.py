import matplotlib.pyplot as plt
import numpy as np
import torch
import os


class Evaluate():
    
    def __init__(self, data_type, exp_type, test_ratio):
        self.path = 'result/' + data_type + '/' + exp_type+ '/'
        self.test_ratio = test_ratio
        self.data_type = data_type
        self.exp_type = exp_type
        self.best = 1
        i = 0
        while True:
            folder = os.path.exists(self.path + 'exp'+ str(i) + '/')
            if folder is False:
                self.path = self.path + 'exp'+ str(i) + '/'
                os.makedirs(self.path)
                break
            i += 1
    
    def record_param_setting(self, window_size, stride, batch_size, learning_rate, weight_decay, model):   
        config = open(self.path+'config.txt', mode='w')
        config.write('data setting:\n'+'    window_size='+str(window_size)+', stride='+str(stride)+'\n')
        config.write('trainning setting:\n'+'   batch_size='+str(batch_size)+', learning_rate='+str(learning_rate)+', weight_decay='+str(weight_decay)+'\n')
        config.write('model setting:\n')
        print(model, file=config)
        config.close()
        
    def visualize(self, train_loss, vaild_loss, model, test_loss):
        index = np.arange(len(train_loss))
        index_ = np.arange(0, len(train_loss), self.test_ratio)
        if test_loss is None:
            if train_loss[-1] < self.best:
                self.best = train_loss[-1]
                torch.save(model, self.path+'model'+'.pkl')
        else:
            if vaild_loss[-1] < self.best:
                self.best = vaild_loss[-1]
                torch.save(model, self.path+'model'+'.pkl')
            plt.plot(index_, test_loss, 'r', linewidth=1.5, label="test")
        plt.grid(color='#7d7f7c', linestyle='-.')
        plt.plot(index, train_loss, 'c--', linewidth=1.5, label="train")
        plt.plot(index, vaild_loss, '2b--', linewidth=1.5, label="vaild")
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.ylim(0, 1e-2)
        plt.legend(loc=1)
        plt.savefig(self.path + 'loss.jpg', dpi=300)
        plt.clf()
        # plt.grid(color='#7d7f7c', linestyle='-.')
        # plt.plot(index_, test_loss, '2r--', linewidth=1.5, label="vaild")
        # plt.title('Loss')
        # plt.xlabel('epoch')
        # plt.ylabel('MSE')
        # plt.ylim(0, 1e-2)
        # plt.legend(loc=1)
        # plt.savefig(self.path + 'test_loss.jpg', dpi=300)
        # plt.clf()

    def draw(self, title, x):
        index = np.arange(len(x))
        plt.grid(color='#7d7f7c', linestyle='-.')
        plt.plot(index, x, 'c', linewidth=1.5, label=title)
        plt.title(title)
        plt.xlabel('epoch')
        plt.legend(loc=1)
        plt.savefig(self.path + title+ '.jpg', dpi=300)
        plt.clf()
        
if __name__ == '__main__':
    evaluate = Evaluate('0degree', 'SFFN')
    model = None
    evaluate.record_param_setting(100, 10, 32, 1e-4, 1e-5, model)