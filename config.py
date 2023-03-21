import argparse
import torch
import lib
import numpy as np
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=100, type=int) #Literature uses 100 / 1000 --> better is 100
parser.add_argument('--num_layers', default=3, type=int) #1 hidden layer
parser.add_argument('--batch_size', default=50, type=int) #50 in first paper and 32 in second paper
parser.add_argument('--dropout_input', default=0, type=float) #0.5 for TOP and 0.3 for BPR
parser.add_argument('--dropout_hidden', default=0.5, type=float) #0.5 for TOP and 0.3 for BPR
parser.add_argument('--n_epochs', default=5, type=int) #number of epochs (10 in literature)
parser.add_argument('--k_eval', default=20, type=int) #value of K durig Recall and MRR Evaluation
# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str) #Optimizer --> Adagrad is the best according to literature
parser.add_argument('--final_act', default='tanh', type=str) #Final Activation Function
parser.add_argument('--lr', default=0.01, type=float) #learning rate (Best according to literature 0.01 to 0.05)
parser.add_argument('--weight_decay', default=0, type=float) #no weight decay
parser.add_argument('--momentum', default=0, type=float) #no momentum
parser.add_argument('--eps', default=1e-6, type=float) #not used
parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting
parser.add_argument("-sigma", type=float, default=None, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]") # weight initialization [-sigma sigma] in literature

####### TODO: discover this ###########
parser.add_argument("--embedding_dim", type=int, default=-1, help="using embedding") 
####### TODO: discover this ###########

# parse the loss type
parser.add_argument('--loss_type', default='TOP1-max', type=str) #type of loss function TOP1 / BPR / TOP1-max / BPR-max
# etc
parser.add_argument('--time_sort', default=False, type=bool) #In case items are not sorted by time stamp
parser.add_argument('--model_name', default='GRU4REC-CrossEntropy', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='../Dataset/RecSys_Dataset_After/', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument("--is_eval", action='store_true') #should be used during testing and eliminated during training
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.cuda:
    torch.cuda.manual_seed(args.seed)