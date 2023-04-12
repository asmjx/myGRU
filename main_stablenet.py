import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm 
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
from utilis.matrix import accuracy
from training.reweighting import weight_learner
import models
from ops.config import parser
from training.schedule import lr_setter
import train_model

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](args=args)

    num_ftrs = model.fc1.in_features
    model.fc1 = nn.Linear(num_ftrs, args.classes_num)# 偷偷改了模型的最后的全连接层
    nn.init.xavier_uniform_(model.fc1.weight, .1)
    nn.init.constant_(model.fc1.bias, 0.)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    train_model_place = train_model(model,torch.device("cuda"),args)
    train_model_place.train()
    # for epoch in range(args.start_epoch, args.epochs):
    #     lr_setter(optimizer, epoch, args)
    #     # train Start###################################
    #     model.train()
    #     for i, (images, target) in enumerate(train_loader):
    #         target = target.cuda(args.gpu, non_blocking=True)
    #         output, cfeatures = model(images) # cfeatures:[batch,512] ->  [128,512]
    #         pre_features = model.pre_features # [n_feature,feature_dim] -> [128,512]
    #         pre_weight1 = model.pre_weight1   # [n_feature,1] -> [128,1]

    #         if epoch >= args.epochp:
    #             weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)
    #         else:
    #             # weight1 [batch,1] ; cfeatures [batch,?]
    #             weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

    #         model.pre_features.data.copy_(pre_features)
    #         model.pre_weight1.data.copy_(pre_weight1)
    #                                         #[1*batch] @ [batch,1] = [1*1] -> variable
    #         loss = criterion(output, target).view(1, -1).mm(weight1).view(1)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()



if __name__ == '__main__':
    main()
