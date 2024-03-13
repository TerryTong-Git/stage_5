from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GNN import GCN
from code.stage_1_code.Result_Saver import Result_Saver
from code.stage_5_code.GNN_Setting import GNN_Setting
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch
from torch.nn import DataParallel
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('citeseer', 'citeseer')
    data_obj.dataset_name='citeseer'
    data_obj.dataset_source_folder_path = './data/stage_5_data/citeseer'  
    adj, features, labels, idx_train, idx_val,idx_test= data_obj.load()
    method_obj = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
    
    if args.cuda:
        method_obj.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
    setting_obj = GNN_Setting('GNN', '')
    
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_5_result/GNN'
    result_obj.result_destination_file_name = 'prediction_result'
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    f1= setting_obj.load_run_save_evaluate(features, adj, labels, idx_train, idx_val, idx_test,args)
    print('************ Overall Performance ************')
    print('RNN f1: ' + str(f1))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    