import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
import os
import sys

class SoftmaxClassifier(nn.Module):
    def __init__(self, n_embedding, softmax_output, n_classes, dropout_rate):
        super(SoftmaxClassifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(n_embedding, softmax_output),
            nn.BatchNorm1d(num_features=softmax_output),
            nn.ELU(),
            nn.Dropout(p=dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(softmax_output, n_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        if x.is_cuda != True:
            x = x.cuda()

        sm1 = self.fc1(x)
        sm2 = self.fc2(sm1)

        return sm2

class moBRCAnet(nn.Module):
    def __init__(self, data, output_size, n_features, n_embedding, dropout_rate):
        super(moBRCAnet, self).__init__()
        self.n_features = n_features
        self.embedding_vector = torch.FloatTensor(n_features, n_embedding).uniform_(0, 1)

        self.data = torch.from_numpy(data)
        if self.data.is_cuda != True:
             self.data = self.data.cuda()
        if self.embedding_vector.is_cuda != True:
             self.embedding_vector = self.embedding_vector.cuda()

        self.f_e = torch.mul(self.data[:, :, None], self.embedding_vector)
        
        self.fc_bn_fx = nn.Sequential(
            nn.Linear(n_embedding, output_size), # 128, 64
            nn.BatchNorm1d(num_features=969), 
            nn.Tanh(),
            nn.Dropout(p=dropout_rate)
        )
        self.fc_bn_fa_1 = nn.Sequential(
            nn.Linear(n_embedding, n_features), # 128, 969
            nn.BatchNorm1d(num_features=n_features),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate)
        )
        self.fc_bn_fa_2 = nn.Sequential(
            nn.Linear(n_features, 1), # f_a_1
            nn.BatchNorm1d(num_features=969)
        )
    

    def forward(self, x):
        if x.is_cuda != True:
            x = x.cuda()

        f_e = torch.mul(x[:, :, None], self.embedding_vector)
        
        f_x = self.fc_bn_fx(f_e)
        f_a_1 = self.fc_bn_fa_1(f_e)
        f_a_2 = self.fc_bn_fa_2(f_a_1)
        f_a_2 = f_a_2.view(-1, self.n_features)

        importance = F.softmax(f_a_2, dim=1)
        new_representation = importance.view(-1, self.n_features, 1) * f_x
        new_representation = torch.sum(new_representation, dim=1)

        return new_representation, importance
        
    