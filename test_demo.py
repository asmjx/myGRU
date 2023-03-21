import scipy.io as sio
import numpy as np
import torch
from torch import nn
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd
from data_phm import DataSet
from torch.autograd import Variable
import torch.utils.data as Data
from collections import OrderedDict
import time
from myGRU import GRU
from config import args

def test_bench(acc_dir = "Data/PHM/Test_set/Bearing2_3/acc_00020.csv",model_dir =  "model\LSTM_net.pkl"):
    ''''''
    dataset = DataSet.load_dataset("phm_data")
    model = torch.load('model/LSTM_net.pkl') 
    data,rul = dataset.get_data(["Bearing1_1"])
    l,r = 2000,3000
    data =  data[l:r]
    rul  =  rul [l:r]
    data,rul = np.array(data),np.array(rul)
    data,rul = torch.tensor(data),torch.tensor(rul).float()
    rul_hat = model(data).cpu().tolist()
    for i in range(len(rul_hat)):
        print("估计值:{}   实际值:{}".format(rul_hat[i],rul[i]))
    plt.figure()
    plt.plot(list(range(len(data))), rul, 'b', label='y_trn')
    plt.plot(list(range(len(data))), rul_hat, 'r', label='pre_trn')
    plt.ylim((0,max (max(rul_hat),max(rul)) + 100 ))
    plt.show()
if __name__ == "__main__":
    test_bench()
