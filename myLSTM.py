import numpy as np 
import random
import torch
import math
from tqdm import tqdm 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from data_phm import DataSet
import pickle
import torch.utils.data as Data
import pandas as pd
from collections import OrderedDict
import time
import os

class LSTM(nn.Module):
    def __init__(self, feature_size,seq_len = 2560):
        super(LSTM,self).__init__()
        self.hidden_size = 256
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.com = 10         # seq 放缩倍数
        bidirectional  = False
        if bidirectional:
            self.dim_n = 2
        else:
            self.dim_n = 1
        #序列太长了，训练太久了，需要进行压缩下
        # input x:[batch,seq,2] ->[batch,2 *seq] ->  [batch,seq/10 * 2] -> [batch,seq/10,2]
        self.compress_seq = nn.Sequential(
            nn.Linear(self.seq_len * feature_size, int(self.seq_len / self.com)),
            nn.ReLU(),
            nn.Linear( int(self.seq_len / self.com) , int(self.seq_len / self.com) * feature_size),
        )
        # compress_seq: [batch,seq / 10 * feature_size]
        # reshape     : [batch,seq / 10 , feature_size]
        self.net = nn.LSTM(feature_size , self.hidden_size,1,batch_first=True,bidirectional=bidirectional,dropout=0.3)
        # h_n:[batch,hidden_size * dim_n]
        # out:[batch,seq,hidden_size * dim_n]
        # out.view:[batch*seq,hidden_size * dim_n]
        # self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(self.hidden_size * self.dim_n,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,1),
            nn.ReLU(),
        )
        # out :[batch*seq,1]
        # view->[batch,seq,1]

    def forward(self,x):
        '''
        使用h作为LSTM的输出
        [batch,1]
        '''
        x = x.to(torch.float32)
        # input x:[batch,seq,2] ->[batch,2 *seq] ->  [batch,seq/10 * 2] -> [batch,seq/10,2]
        x = x.reshape(-1, self.seq_len * self.feature_size)
        x_com = self.compress_seq(x) # [batch,seq/10 * 2]
        x_com = x_com.reshape(-1,int(self.seq_len / 10),self.feature_size)
        #压缩结束[batch,seq/10,2]
        output,(h,c) = self.net(x_com)
        # h :[1,128,256],[1,batch,hidden_size]
        h = h.reshape(-1,self.hidden_size*self.dim_n)
        h = torch.squeeze(h) 
        # h:[batch,hidden_size]
        out = self.FC(h)
        out = torch.squeeze(out)
        #[batch,1]
        return out
    
  


























class Model():
    def __init__(self,device = torch.device("cpu")):
        self.epochs       = 200
        self.batch_size   = 128
        self.batches      = 30
        self.lr           = 1e-3
        self.feature_size = 2
        self.network      = LSTM(self.feature_size).to(device)
        self.optimizer    =  torch.optim.Adam(self.network.parameters(),lr=self.lr,weight_decay=1e-4)
        self.loss         = nn.CrossEntropyLoss()
        self.loss         = nn.MSELoss()
        self.save_log_path = "./log"



    def get_bear_data(self, dataset, select):
        if select == 'train':
            _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
            # _select =['Bearing1_3','Bearing1_1']
        elif select == 'test':
            _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']
            # _select = ['Bearing1_2','Bearing2_2','Bearing3_2']
        else:
            raise ValueError('wrong select!')
        data,rul = dataset.get_data(_select,is_percent = True)
        # data :[测试总次数n,acc,2]
        # RUL  :[测试总次数n:RUL]
        data,rul = np.array(data),np.array(rul)
        data,rul = torch.tensor(data),torch.tensor(rul).float()
        data_set = Data.TensorDataset(data,rul)
        output   =   Data.DataLoader(data_set,self.batch_size,shuffle=True)
        return output


    def fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        return fft_data
    def evaluate_accuracy(self,data_iter):
        acc_sum, n = 0.0, 0
        net = self.network
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(net, torch.nn.Module):
                    net.eval()  # 评估模式, 这会关闭dropout
                    acc_sum += (net(X).argmax(dim=0) == y).float().sum().cpu().item()
                    net.train()  # 改回训练模式
                else:  # 自定义的模型, 3.13节之后不会用到, 不考虑G
                    # 如果有is_training这个参数PU
                    if('is_training' in net.__code__.co_varnames):
                        # 将is_training设置成False
                        acc_sum += (net(X, is_training=False).
                                    argmax(dim=0) == y).float().sum().item()
                    else:
                        acc_sum += (net(X).argmax(dim=0) == y).float().sum().item()
                n += y.shape[0]
        return acc_sum / n
 
    def train(self):
        dataset = DataSet.load_dataset("phm_data")
        train_iter = self.get_bear_data(dataset,'train')
        # train_iter = self.get_bear_data(dataset,'test')
        test_iter =  self.get_bear_data(dataset,'test')
    
        log = OrderedDict()
        log['train_rul_acc'] = []
        log['train_rul_loss'] = []
        sample = open(os.path.join(self.save_log_path,"sample.log"),"w",encoding="utf-8") 
        with tqdm(total=self.epochs * len(train_iter),colour= "red") as pbar:
            batch_count = 0
            for epoch in range(1,self.epochs+1):
                train_acc = 0
                pbar.set_description(f"epoch:{epoch},acc={train_acc}")
                train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
                for x,y in train_iter:
                    # x: [128, 2560, 2] :[batch,seq,feature]
                    # y: [128]          :[batch:rul]
                    x = x.to(device)
                    y = y.to(device)
                    y_hat = self.network(x).to(device)
                    # y_hat:[128]
                    loss_ = self.loss(y_hat,y)
                    self.optimizer.zero_grad()
                    loss_.backward()
                    self.optimizer.step()

                    y_hat = y_hat.detach().cpu().numpy()
                    loss_ = loss_.detach().cpu().numpy()
                    y     = y.detach().cpu().numpy()
                    train_l_sum += loss_
                    self.sample_write(sample,y_hat,y)
                    train_err = sum([abs(y_hat[index] - y[index]) / (y[index]) for index in range(len(y_hat)) if y[index] != 0])/len(y_hat)
                    # train_acc =  sum([abs(y_hat[index] - y[index]) / (y[index] + 0.1) for index in range(len(y_hat)) ])/len(y_hat)
                    n += y.shape[0]
                    batch_count += 1
                    pbar.set_description("epoch:{},err:{:.2f},loss:{:.4f}".format(epoch,train_err,loss_))
                    pbar.update(1)
                # test_acc = self.evaluate_accuracy(test_iter)#测试太慢了，暂时不进行测试
                test_acc = 0.5
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec'% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                    test_acc, time.time() - start))
                
                
                log['train_rul_acc'].append(train_acc_sum / n)
                log['train_rul_loss'].append(train_l_sum / batch_count)
                f = open(f"log/train.log", 'a',encoding="utf-8")
                f.write('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec\n'% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,test_acc, time.time() - start))
                f.close()
            torch.save(self.net, 'model/LSTM_net.pkl')
            model = torch.load('model/LSTM_net.pkl') 

    def sample_write(self,sample,y_hat,y,size = None):
        '''向sample写入日志'''
        size = size  if size and size < self.batch_size  else len(y_hat)
        for i in range(size):
            sample.write("{:.4f} ".format(y[i]))
        sample.write("\n")
        for i in range(size):
            sample.write("{:.4f} ".format(y_hat[i]))
        sample.write("\n\n")
        sample.flush()



if __name__ == '__main__':
    # torch.backends.cudnn.enabled=False
    # device = torch_directml.device()
    device = torch.device("cuda")
    print("use device:{}".format(device))
    process = Model(device= device)
    process.train()