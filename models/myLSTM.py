import numpy as np 
import torch 
from torch import nn

class LSTM_stable(nn.Module):
    def __init__(self,seq_len = 2560,args = None):
        super(LSTM_stable,self).__init__()
        self.hidden_size = args.feature_dim #这是一个需要特别注意的参数，因为 flatten_featuers [B,hid_size]需要和c_feature的size一样才行
                                #cfeature 的size 在config文件里面定义了[n_feature,feature_dim]->[128,512]参见本文件【42】行
        self.seq_len = seq_len
        self.feature_size = args.feature_size
        self.com = 10         # seq 放缩倍数
        bidirectional  = False
        if bidirectional:
            self.dim_n = 2
        else:
            self.dim_n = 1
        #序列太长了，训练太久了，需要进行压缩下
        # input x:[batch,seq,2] ->[batch,2 *seq] ->  [batch,seq/10 * 2] -> [batch,seq/10,2]
        self.compress_seq = nn.Sequential(
            nn.Linear(self.seq_len * self.feature_size, int(self.seq_len / self.com)),
            nn.ReLU(),
            nn.Linear( int(self.seq_len / self.com) , int(self.seq_len / self.com) * self.feature_size),
        )
        # compress_seq: [batch,seq / 10 * feature_size]
        # reshape     : [batch,seq / 10 , feature_size]
        self.net = nn.LSTM(self.feature_size , self.hidden_size,1,batch_first=True,bidirectional=bidirectional,dropout=0.3)
        # h_n:[batch,hidden_size * dim_n]
        # out:[batch,seq,hidden_size * dim_n]
        # out.view:[batch*seq,hidden_size * dim_n]
        # self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(self.hidden_size * self.dim_n,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,1),
            nn.LeakyReLU(),
        )
        # out :[batch*seq,1]
        # view->[batch,seq,1]
        # 在此处添加论文中所指的全局信息保存的模块,保存训练中的prefeatures\pre_weight1
        self.register_buffer('pre_features', torch.zeros(args.n_feature, args.feature_dim))
        self.register_buffer('pre_weight1', torch.ones(args.n_feature, 1))
        if args.n_levels > 1:
            self.register_buffer('pre_features_2', torch.zeros(args.n_feature, args.feature_dim))
            self.register_buffer('pre_weight1_2', torch.ones(args.n_feature, 1))
        if args.n_levels > 2:
            self.register_buffer('pre_features_3', torch.zeros(args.n_feature, args.feature_dim))
            self.register_buffer('pre_weight1_3', torch.ones(args.n_feature, 1))
        if args.n_levels > 3:
            self.register_buffer('pre_features_4', torch.zeros(args.n_feature, args.feature_dim))
            self.register_buffer('pre_weight1_4', torch.ones(args.n_feature, 1))
        if args.n_levels > 4:
            print('WARNING: THE NUMBER OF LEVELS CAN NOT BE BIGGER THAN 4')

    def forward(self,x):
        '''
        使用h作为LSTM的输出
        [batch,1]
        '''
        x = x.to(torch.float32)
        # input x:[batch,seq,2] ->[batch,2 *seq] ->  [batch,seq/10 * 2] -> [batch,seq/10,2]
        x = x.reshape(-1, self.seq_len * self.feature_size)
        x_com = self.compress_seq(x) # [batch,seq/10 * 2]
        x_com = x_com.reshape(-1,int(self.seq_len / self.com),self.feature_size)
        #压缩结束[batch,seq/10,2]
        output,(h,c) = self.net(x_com)
        # h :[1,128,256],[1,batch,hidden_size]
        h = h.reshape(-1,self.hidden_size*self.dim_n)
        h = torch.squeeze(h) 
        # h:[batch,hidden_size]
        flatten_featuers = torch.flatten(h,1)
        out = self.FC(h)
        out = torch.squeeze(out)
        #[batch,1]
        return out,flatten_featuers
    
def LSTM_with_table( seq_len = 2560,args = None):
    '''
    '''
    return LSTM_stable( seq_len = seq_len,args = args)
