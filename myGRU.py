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
# from config import args
# Define GRU Neural Networks
class GRU(nn.Module):
    """
        Parameters:
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, num_layers=1,seq_len = 2560):
        super().__init__()
        self.hidden_size = 256
        self.seq_len = seq_len
        self.feature_size = input_size
        bidirectional  = False # 是否是双向网络
        self.com = 10         # seq 放缩倍数
        if bidirectional:
            self.dim_n = 2
        else:
            self.dim_n = 1
        self.compress_seq = nn.Sequential(
            nn.Linear(self.seq_len * self.feature_size, int(self.seq_len / self.com)),
            nn.ReLU(),
            nn.Linear( int(self.seq_len / self.com) , int(self.seq_len / self.com) * self.feature_size),
        )
        self.net = nn.GRU(input_size, self.hidden_size, batch_first=True,num_layers = 1)  # utilize the GRU model in torch.nn
        self.FC = nn.Sequential(
            nn.Linear(self.hidden_size * self.dim_n,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,1),
            nn.ReLU()                        #此处的问题，为什么添加了ReLU或者其他的激活函数作为输出后，预测的结果之间全是0(tanh为负数)
        ) 
    def forward(self,x):
        x = x.to(torch.float32)                                           # x: [batch,seq , 2 ] 
        x = x.reshape(-1, self.seq_len * self.feature_size)               # x: [batch,2 * seq ]
        x_com = self.compress_seq(x)                                      # x: [batch,seq/10*2]
        x_com = x_com.reshape(-1,int(self.seq_len / self.com),self.feature_size)# x: [batch,seq/10,2]
        x, h = self.net(x_com)                                            # x: [batch,seq/10,hid]
        h = h.reshape(-1,self.hidden_size*self.dim_n)                     # h: [1 , batch, hid]
        h = torch.squeeze(h)                                              # h: [batch,hid]
        out = self.FC(h)
        out = torch.squeeze(out)                                          #out:[batch,1]
        return out
class Model():
    def __init__(self):
        self.epochs       = 200
        self.batch_size   = 128
        self.batches      = 30
        self.lr           = 0.5
        self.feature_size = 2
        self.network      = GRU(self.feature_size)
        self.optimizer    =  torch.optim.Adam(self.network.parameters(),lr=self.lr,weight_decay=1e-4)
        self.loss         = nn.CrossEntropyLoss()
        self.loss         = nn.MSELoss()
        # self.loss         = nn.KLDivLoss(reduction='batchmean') # KL 散度可用于衡量不同的连续分布之间的距离, 在连续的输出分布的空间上(离散采样)上进行直接回归时
        lambda1 = lambda epoch: 1 / (epoch + 1) #调整学习率
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])



    def get_bear_data(self, dataset, select):
        if select == 'train':
            # _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
            _select =['Bearing1_3','Bearing1_1']
        elif select == 'test':
            _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']
            # _select = ['Bearing1_2','Bearing2_2','Bearing3_2']
        else:
            raise ValueError('wrong select!')
        data,rul = dataset.get_data(_select)
        # data :[测试总次数n,acc,2]
        # RUL  :[测试总次数n:RUL]
        data,rul = np.array(data),np.array(rul)
        data,rul = torch.tensor(data),torch.tensor(rul).float()
        data_set = Data.TensorDataset(data,rul)
        output   =   Data.DataLoader(data_set,self.batch_size,shuffle=True)
        return output

    def train(self):
        dataset = DataSet.load_dataset("phm_data")
        train_iter = self.get_bear_data(dataset,'train')
        # train_iter = self.get_bear_data(dataset,'test')
        test_iter =  self.get_bear_data(dataset,'test')
    
        log = OrderedDict()
        log['train_rul_acc'] = []
        log['train_rul_loss'] = []
        dif = 1 #预测精度 0.01 相差小于 0.1 被认为预测成功
        f_pro = open("./log/pro_log.log","w",encoding="utf-8") 
        with tqdm(total=self.epochs * len(train_iter),colour= "red") as pbar:
            batch_count = 0
            for epoch in range(1,self.epochs+1):
                train_acc = 0
                train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
                for x,y in train_iter:
                    # x: [128, 2560, 2] :[batch,seq,feature]
                    # y: [128]          :[batch:rul]
                    y_hat = self.network(x)
                    # y_hat:[128]
                    loss_ = self.loss(y_hat,y)
                    self.optimizer.zero_grad()
                    loss_.backward()
                    self.optimizer.step()
                    train_l_sum += loss_.cpu().item()
                    f_pro.write("{}\n".format(  list(map(int,y_hat.cpu().tolist()[0:10] ) )    ))
                    f_pro.write("{}\n\n".format(list (map(int,y.tolist()[0:10]         ))     ))
                    f_pro.flush()
                    train_acc = 1 - sum([abs(y_hat[index] - y[index]) / (y[index]) for index in range(len(y_hat)) if y[index] != 0])/len(y_hat)
                    # train_std = sum([abs(y_hat[index] - y[index]) for index in range(len(y_hat)) if y[index] != 0])/len(y_hat)
                    n += y.shape[0]
                    batch_count += 1
                    pbar.set_description("epoch:{},acc={:.2f}%".format(epoch,train_acc * 100))
                    pbar.update(1)
                self.scheduler.step()
                # test_acc = self.evaluate_accuracy(test_iter)#测试太慢了，暂时不进行测试
                test_acc = 0.5
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec'% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                    test_acc, time.time() - start))
                
                log['train_rul_acc'].append(train_acc_sum / n)
                log['train_rul_loss'].append(train_l_sum / batch_count)
                f = open(f"log/train.log", 'a',encoding="utf-8")
                f.write('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec\n'% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,test_acc, time.time() - start))
                f.close()
            torch.save(self.network, 'model/LSTM_net.pkl')
            model = torch.load('model/LSTM_net.pkl') 

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
if __name__ == '__main__':
    torch.backends.cudnn.enabled=False
    process = Model()
    process.train()


if 0:
# checking if GPU is available
    device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    # 数据读取&类型转换
    data_x = np.array(pd.read_csv('Data_x.csv', header=None)).astype('float32')
    data_y = np.array(pd.read_csv('Data_y.csv', header=None)).astype('float32')

    # 数据集分割
    data_len = len(data_x)
    t = np.linspace(0, data_len, data_len + 1)

    train_data_ratio = 0.8  # Choose 80% of the data for training
    train_data_len = int(data_len * train_data_ratio)

    train_x = data_x[5:train_data_len]
    train_y = data_y[5:train_data_len]
    t_for_training = t[5:train_data_len]

    test_x = data_x[train_data_len:]
    test_y = data_y[train_data_len:]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    INPUT_FEATURES_NUM = 5
    OUTPUT_FEATURES_NUM = 1
    train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 1
    train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1

    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    gru_model = GRU(INPUT_FEATURES_NUM, 30, output_size=OUTPUT_FEATURES_NUM, num_layers=1)  # 30 hidden units
    print('GRU model:', gru_model )
    print('model.parameters:', gru_model .parameters)
    print('train x tensor dimension:', Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gru_model .parameters(), lr=1e-2)

    prev_loss = 1000
    max_epochs = 2000

    train_x_tensor = train_x_tensor.to(device)

    for epoch in range(max_epochs):
        output = gru_model(train_x_tensor).to(device)
        loss = criterion(output, train_y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < prev_loss:
            torch.save(gru_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
            prev_loss = loss

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # prediction on training dataset
    pred_y_for_train = gru_model(train_x_tensor).to(device)
    pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- test -------------------
    gru_model = gru_model .eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 1,
                                    INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = gru_model(test_x_tensor).to(device)
    pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
    print("test loss：", loss.item())

    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_y, 'b', label='y_trn')
    plt.plot(t_for_training, pred_y_for_train, 'y--', label='pre_trn')

    plt.plot(t_for_testing, test_y, 'k', label='y_tst')
    plt.plot(t_for_testing, pred_y_for_test, 'm--', label='pre_tst')

    plt.xlabel('t')
    plt.ylabel('Vce')
    plt.show()