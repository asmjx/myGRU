import numpy as np
import torch
from torch import nn
from tqdm import tqdm 
import matplotlib.pyplot as plt
from utilis.data_phm import DataSet
from torch.autograd import Variable
import torch.utils.data as Data
from collections import OrderedDict
import matplotlib.pyplot as plt
from training.reweighting import weight_learner
from training.schedule import lr_setter
import os
    
class Go_training():
    def __init__(self,model,device,args):
        self.args = args
        self.epochs = args.epochs
        self.batch_size   = args.batch_size#128
        self.lr           = args.lr       #0.001
        self.feature_size = 2
        self.device = device
        self.network      = model.to(device)
        self.optimizer    =  torch.optim.Adam(self.network.parameters(),lr=self.lr,weight_decay=1e-4)
        # self.optimizer    =  torch.optim.SGD(self.network.parameters(),lr=self.lr,weight_decay=1e-4,momentum=0.78)
        # self.loss         = nn.CrossEntropyLoss()
        self.loss         = nn.MSELoss()
        self.save_log_path = "./log"
        # self.loss         = nn.KLDivLoss(reduction='batchmean') # KL 散度可用于衡量不同的连续分布之间的距离, 在连续的输出分布的空间上(离散采样)上进行直接回归时
        # lambda1 = lambda epoch: 1 / (epoch + 1) #调整学习率
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])

        # 统计部分
        self.log = OrderedDict()
        self.log['train_err'] = []
        self.log['test_err'] = []
        self.log['train_loss'] = []
        self.log['test_loss'] = []
        # sample<记录每一次的预测结果的文件>
        self.sample = open(os.path.join(self.save_log_path,"sample.log"),"w",encoding="utf-8") 
        self.initLogF()

    def train(self):
        dataset = DataSet.load_dataset("phm_data")
        train_iter = self.get_bear_data(dataset,'train')
        test_iter =  self.get_bear_data(dataset,'test')
    
        with tqdm(total=self.epochs * len(train_iter),colour= "red") as pbar:
            for epoch in range(0,self.epochs):
                lr_setter(self.optimizer,epoch, self.args)
                tr_log = {"train_l_sum":0,"train_err_sum":0,"batch_count":0,"test_loss":0,"test_err":0}#训练的过程记录数据

                self.fit(tr_log,train_iter,pbar,epoch)

                tr_log["test_err"],tr_log["test_loss"] = self.evaluate_accuracy(test_iter)#测试太慢了，暂时不进行测试
                epoch_info = f'epoch{epoch + 1}, train err:{tr_log["train_err_sum"]/tr_log["batch_count"] :.4f},\
                        test err:{tr_log["test_err"]:.4f},train_loss:{tr_log["train_l_sum"] / tr_log["batch_count"]:.4f},test_loss:{tr_log["test_loss"]:.4f}\n'
                print(epoch_info)
                self.do_save_log(epoch_info,**tr_log)
            #训练结束后保存图像
            self.paint(self.log)
            self.save_model() 

    def fit(self,tr_log,train_iter,pbar,epoch):
        self.network.train()
        for i,(x,y) in enumerate(train_iter):
            x = x.to(self.device)
            y = y.to(self.device)
            # x: [128, 2560, 2] :[batch,seq,feature]
            # y: [128]          :[batch:rul]
            y_hat,cfeatures = self.network(x) # y_hat:[128]  cfearture[B:hid]
            pre_features = self.network.pre_features # [n_feature,feature_dim] -> [128,512]
            pre_weight1 = self.network.pre_weight1   # [n_feature,1] -> [128,1]

            if epoch >= self.args.epochp:
                weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, self.args, epoch, i)
            else:
                # weight1 [batch,1] ; cfeatures [batch,?]
                weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

            self.network.pre_features.data.copy_(pre_features)
            self.network.pre_weight1.data.copy_(pre_weight1)
                        #[1*batch] @ [batch,1] = [1*1] -> variable
            loss_ = self.loss(y_hat, y).view(1, -1).mm(weight1).view(1)

            self.optimizer.zero_grad()
            loss_.backward()
            self.optimizer.step()

            # 统计部分$$$$$$$$$$
            y_hat ,loss_ ,y= y_hat.detach().cpu().numpy(), loss_.detach().cpu().numpy(),y.detach().cpu().numpy()
            tr_log["train_l_sum"] += loss_
            self.sample_write(self.sample,y_hat,y)
            tr_log["train_err"] = sum([abs(y_hat[index] - y[index]) / (y[index]) for index in range(len(y_hat)) if y[index] != 0])/len(y_hat)
            tr_log["train_err_sum"] += tr_log["train_err"]
            tr_log["batch_count"] += 1
            pbar.set_description("epoch:{},err:{:.2f},loss:{:.4f}".format(epoch,tr_log["train_err"],loss_))
            pbar.update(1)
            #end
        


    def get_bear_data(self, dataset, select):
        if select == 'train':
            _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
            # _select =['Bearing1_3','Bearing1_1']
            # _select =['Bearing1_1']
        elif select == 'test':
            _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']
            # _select = ['Bearing1_2','Bearing2_2','Bearing3_2']
        else:
            raise ValueError('wrong select!')
        data,rul = dataset.get_data(_select,is_percent = False)
        # data :[测试总次数n,acc,2]
        # RUL  :[测试总次数n:RUL]
        data,rul = np.array(data),np.array(rul)
        data,rul = torch.tensor(data),torch.tensor(rul).float()
        data_set = Data.TensorDataset(data,rul)
        output   =   Data.DataLoader(data_set,self.batch_size,shuffle = True)
        return output

    def initLogF(self):#用来初始化存放log的文件夹
        if  not os.path.exists(self.save_log_path):
            os.mkdir(self.save_log_path)
        f1,f2 =  open(os.path.join(self.save_log_path,"train.log"),'w'), open(os.path.join(self.save_log_path,"sample.log"),'w') 
        f1.close()
        f2.close()

    def do_save_log(self,epoch_info,train_err_sum,test_err,train_l_sum,test_loss,batch_count):
        '''保存一下log信息'''
        self.log['train_err'].append(train_err_sum/batch_count)
        self.log['test_err'].append(test_err)
        self.log['train_loss'].append(train_l_sum/batch_count)
        self.log['test_loss'].append(test_loss)
        f = open(os.path.join(self.save_log_path,"train.log"), 'a',encoding="utf-8")
        f.write(epoch_info + "\n")
        f.close()
    
    def save_model(self):
        "保存模型"
        torch.save(self.network.cpu(), 'model/GRU_net.pkl')
        model = torch.load('model/GRU_net.pkl') 


    def evaluate_accuracy(self,data_iter):
        test_err_sum = 0
        test_loss_sum = 0
        net = self.network
        batch_count = 0
        with torch.no_grad():
            for x, y in data_iter:
                x = x.to(self.device)
                y = y.to(self.device)

                net.eval()  # 评估模式, 这会关闭dropout
                y_hat = net(x).to(self.device)
                loss = self.loss(y_hat,y)
                net.train()  # 改回训练模式

                test_loss_sum += loss.cpu().item()
                y_hat = y_hat.detach().cpu().numpy()
                y     = y.detach().cpu().numpy()
                test_err = sum([abs(y_hat[index] - y[index]) / (y[index]) for index in range(len(y_hat)) if y[index] != 0])/len(y_hat)
                test_err_sum += test_err
                batch_count += 1
        return test_err_sum/batch_count, test_loss_sum/batch_count
    
    def paint(self,log:OrderedDict):
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        ax1 = fig1.subplots() 
        ax2 = fig2.subplots()
        ax1.plot(log['train_err'], label="train_err")
        ax1.plot(log['test_err'],label = "test_err")
        ax2.plot(log['train_loss'], label="train_loss")
        ax2.plot(log['test_loss'],label = "test_loss")
        ax1.set_title("err")
        ax2.set_title("loss")
        ax1.legend()
        ax2.legend()
        # plt.show()
        fig1.savefig(os.path.join(self.save_log_path,"err.png"))
        fig2.savefig(os.path.join(self.save_log_path,"loss.png"))

    
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
