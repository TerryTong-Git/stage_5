'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchtext
import numpy as np
import wandb
import random
import pandas as pd


from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if input.size() == output.size():
            output = F.relu(output + input)  # Assuming activation function is ReLU
        else:
            # If sizes don't match, you might need to project input to match output size
            # Here we'll use a simple linear layer to project input to output size
            projection = nn.Linear(input.size(1), output.size(1)).cuda()
            output = F.relu(output + projection(input))
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output

    def __init__(self, nfeat, nhid, nclass, dropout):
        method.__init__(self)
        nn.Module.__init__(self)
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def train1(self):
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="GCN_Res",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.0001,
                "epochs": self.max_epoch,
            },
        )
        evaluator =  Evaluate_Accuracy()

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        for epoch in range(self.args.epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            acc_train = evaluator.accuracy(output[self.idx_train], self.labels[self.idx_train])
            f1_train = evaluator.f1_score(output[self.idx_train], self.labels[self.idx_train])
            precision_train = evaluator.precision(output[self.idx_train], self.labels[self.idx_train])
            recall_train = evaluator.recall(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            optimizer.step()

            if not self.args.fastmode:
                self.eval()
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                output = self.forward(self.features, self.adj)

            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
            acc_val =evaluator.accuracy(output[self.idx_val], self.labels[self.idx_val])
            f1_val =evaluator.f1_score(output[self.idx_val], self.labels[self.idx_val])
            precision_val =evaluator.precision(output[self.idx_val], self.labels[self.idx_val])
            recall_val =evaluator.recall(output[self.idx_val], self.labels[self.idx_val])
            if epoch%100==0:
                print('Epoch: {:04d}'.format(epoch+1),
                    "========================================================= \n",
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'loss_val: {:.4f}\n'.format(loss_val.item()),
                    "========================================================= \n",
                    'acc_train: {:.4f}'.format(acc_train),
                    'acc_val: {:.4f}\n'.format(acc_val),
                    "========================================================= \n",
                    'f1_train: {:.4f}'.format(f1_train),
                    'f1_val: {:.4f}\n'.format(f1_val),
                    "========================================================= \n",
                    'precision_train: {:.4f}'.format(precision_train),
                    'precision_val: {:.4f}\n'.format(precision_val),
                    "========================================================= \n",
                    'recall_train: {:.4f}'.format(recall_train),
                    'recall_val: {:.4f}\n'.format(recall_val),
                    "========================================================= \n",
                )
                
            wandb.log({"val_accuracy": acc_val,
                        "val_loss": loss_val.item(),
                        "train_accuracy": acc_train, 
                        "train_loss":loss_train.item(),
                        "train_f1":f1_train,
                        "val_f1":f1_val,
                        "train_precision":precision_train,
                        "val_precision":precision_val,
                        "train_recall": recall_train,
                        "val_recall": recall_val,
                        }
                      )
    def test1(self):
        self.eval()
        evaluator=  Evaluate_Accuracy()
        output = self.forward(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = evaluator.accuracy(output[self.idx_test], self.labels[self.idx_test])
        f1_test = evaluator.f1_score(output[self.idx_test], self.labels[self.idx_test])
        precision_test = evaluator.precision(output[self.idx_test], self.labels[self.idx_test])
        recall_test = evaluator.recall(output[self.idx_test], self.labels[self.idx_test])

        
        print("Test set results:",
            "=========================================================\n",
            "loss= {:.4f} \n".format(loss_test.item()),
            "accuracy= {:.4f} \n".format(acc_test),
            "f1_score= {:.4f} \n".format(f1_test),
            "precision= {:.4f} \n".format(precision_test),
            "recall= {:.4f} \n".format(recall_test),
            "=========================================================\n",
            )

        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return 
    
    def run(self, *argv):
        print('method running...')
        print('--start training...')
        # train_set, val_set = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), int(len(self.dataset)*0.2)])
        # self.train_set = train_set
        # self.test_set = val_set
        self.features = argv[0]
        self.adj = argv[1]
        self.labels = argv[2]
        self.idx_train = argv[3]
        self.idx_val = argv[4]
        self.idx_test = argv[5]
        self.args = argv[6]
        self.train1()

        print('--start testing...')
        self.test1()
        # pred_y = self.test(self.data['test']['X'])
        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
        return