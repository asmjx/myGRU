import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import models
from ops.config import parser
import train_model
from utilis.data_phm import DataSet
def main():
    args = parser.parse_args()

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)
    #     cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](args=args)


    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    train_model_place = train_model.Go_training(model,torch.device("cuda"),args)
    train_model_place.train()

if __name__ == '__main__':
    main()
