import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter

from training.reweighting import weight_learner


def train(train_loader, model, criterion, optimizer, epoch, args):
    ''' TODO write a dict to save previous featrues  check vqvae,
        the size of each feature is 512, os we need a tensor of 1024 * 512
        replace the last one every time
        and a weight with size of 1024,
        replace the last one every time
        TODO init the tensors
    '''

    model.train()
    for i, (images, target) in enumerate(train_loader):
        target = target.cuda(args.gpu, non_blocking=True)
        output, cfeatures = model(images) # cfeatures:[batch,512] ->  [128,512]
        pre_features = model.pre_features # [n_feature,feature_dim] -> [128,512]
        pre_weight1 = model.pre_weight1   # [n_feature,1] -> [128,1]

        if epoch >= args.epochp:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)
        else:
            # weight1 [batch,1] ; cfeatures [batch,?]
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

        model.pre_features.data.copy_(pre_features)
        model.pre_weight1.data.copy_(pre_weight1)
                                        #[1*batch] @ [batch,1] = [1*1] -> variable
        loss = criterion(output, target).view(1, -1).mm(weight1).view(1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

