# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:53 2018
@author: a273
TODO
    should class DataSet only arange, save and load?
"""

import os
import operator
import random
from tqdm import tqdm 
import scipy.io as sio
import pickle as pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
#区间缩放，返回值为缩放到[0, 1]区间的数据
data = ...
# Standard_data=MinMaxScaler().fit_transform(data)

class DataSet(object):
    def __init__(self,name ='phm_data',
                index      =['bearing_name','RUL','quantity','data'],
                save_path  ='data_pkl/',
                load_path = './Data/PHM/'):
        self.name = name
        self.index = index         #[bearname,acc,rul]
        self.dataset = []
        self.save_path = save_path #将自身保存为.pkl文件，以便于后面继续调用
        self.load_path = load_path #加载数据集
        self.each_acc  = 10        #每一个振动序列的时间 s
        self.RUL_dict = {'Bearing1_1':0,'Bearing1_2':0,
                        'Bearing2_1':0,'Bearing2_2':0,
                        'Bearing3_1':0,'Bearing3_2':0,
                        'Bearing1_3':5730,'Bearing1_4':339,'Bearing1_5':1610,'Bearing1_6':1460,'Bearing1_7':7570,
                        'Bearing2_3':7530,'Bearing2_4':1390,'Bearing2_5':3090,'Bearing2_6':1290,'Bearing2_7':580,
                        'Bearing3_3':820}
        self.total_time_RUL = {}                 #轴承的全部寿命长度 [bearname:str : RUL_p:float]
        #用来存储phm数据集的信息,字典:{bearName:[rul1,rul2....]}
        self.info    = {BearName:[] for BearName in self.RUL_dict} #为了方便后面能够给一个bearName和一个acc序列编号，可以方便的给出RUL
        self.make_data()
        # print(self.getRUL(*"Bearing2_3\acc_01194.csv".split('\\') ))

    def make_data(self):
        '''制作数据集'''
        for path_1 in ['Learning_set/','Test_set/']:
            bearings_names = os.listdir(self.load_path + path_1)
            bearings_names.sort()
            for bearings_name in bearings_names:
                file_names = os.listdir(self.load_path + path_1 + bearings_name + '/')
                file_names.sort()
                total_acc_len = len(file_names)
                with tqdm(total= total_acc_len,colour= "red") as pbar:
                    pbar.set_description(f"bear{bearings_name}")
                    total_time = total_acc_len * self.each_acc  + self.RUL_dict[bearings_name] # 获得轴承总体寿命
                    self.total_time_RUL[bearings_name] = total_time # 为这个轴承更新整体寿命
                    for acc_name in file_names:
                        if 'acc' in acc_name:
                            RUL = self._getRUL(bearings_name,acc_name,total_acc_len)
                            # print(f"{bearings_name}:{acc_name}:{RUL}:{percentage_RUL}")
                            df = pd.read_csv(self.load_path  + path_1 + bearings_name + '/'\
                                            + acc_name,header=None)
                            acc = np.array(df.loc[:,4:6])
                            append_item = [bearings_name,acc,RUL]
                            self.info[bearings_name].append(RUL)  # 保存对应的每一个轴承RUL信息
                            self.dataset.append(append_item)
                        pbar.update(1)
                    
        self.save()


    def getRUL(self,BearName:str,acc_name:str)->int:
        acc_idx = int(acc_name.strip("acc_").strip("0").strip(".csv"))
        res = self.info[BearName][acc_idx]
        return res
    
    def _getRUL(self,BearName:str,acc_name:str,total_acc_len:int):
        '''根据轴承名称和轴承序列获得RUL
            acc_name: acc_00001.csv
            total_acc_le:轴承文件夹里面一共有多少acc序列
        '''
        acc_name_substr = acc_name.strip("acc_").strip("0").strip(".csv")
        # print(acc_name_substr,acc_name)
        RUL_time = self.RUL_dict[BearName]
        total_time = total_acc_len * self.each_acc  + RUL_time #轴承总寿命
        RUL = total_time - int(acc_name_substr) * self.each_acc  #
        percentage_RUL = RUL / total_time                #剩余寿命百分比
        return RUL


    def get_data(self,BearNames:list,is_percent = False):
        '''返回对应轴承的所有数据
        input :['Bearing1_3','Bearing1_1']
        return:data:[acc:[2]],label:[rul:int]
        '''
        res_data  = []
        res_label = []
        for item in self.dataset:# item[0]:bearName,item[1]:data(acc),item[2]:label(rul)
            if item[0] in BearNames:
                res_data.append(item[1])
                if is_percent: #如果要返回百分比的话
                    res_label.append(item[2]/self.total_time_RUL[item[0]])
                else:
                    res_label.append(item[2])
        return res_data,res_label
    
    


    def shuffle(self):
        random.shuffle(self.dataset)


    def save(self):
        '''
        保存当前的data_phm,以便于以后加载
        '''
        pickle.dump(self, open(self.save_path + 'DataSet_' +
                                        self.name + '.pkl', 'wb'), True)
        print('dataset ', self.name, ' has benn saved\n')


    @staticmethod
    def load_dataset(name):
        '''
        Load this DataSet with name and default path './data/'.
        
        Args:
            name: The name of DataSet.
        Return:
            DataSet
        '''
        save_path = './data_pkl/'
        full_name = save_path + 'DataSet_' + name + '.pkl'
        load_class = pickle.load(open(full_name,'rb'))
        print('dataset ', name, ' has been load')
        return load_class



if __name__ == '__main__':
    phm = DataSet()
    dataset = DataSet.load_dataset('phm_data')
    print('Generated data_set')