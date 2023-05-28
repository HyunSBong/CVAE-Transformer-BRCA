import math
import numpy as np
import pandas as pd
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

from moBRCAnet_gene_pytorch_model import moBRCAnet, SoftmaxClassifier

import wandb

def load_data(train_x, test_x, train_y, test_y, n_gene):
    X_train = pd.read_csv(train_x, delimiter=",", dtype=np.float32)
    X_test = pd.read_csv(test_x, delimiter=",", dtype=np.float32)
    Y_train = pd.read_csv(train_y, delimiter=",", dtype=np.float32).values
    Y_test = pd.read_csv(test_y, delimiter=",", dtype=np.float32).values

    X_gene_train = X_train.values
    X_gene_test = X_test.values

    n_classes = len(Y_train[1])
    
    dataset = {'train_set': (X_train, Y_train),
               'test_set': (X_test, Y_test),
               'gene_set': (X_gene_train, X_gene_test),
               'n_classes': n_classes
              }
    return dataset

def main(args, dataset):
    wandb.init(project="moBRCAnet gene level pytorch", reinit=True)
    wandb.config.update(args)

    ### GPU 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # Train/Test dataset
    (X_train, Y_train) = dataset['train_set']
    (X_test, Y_test) = dataset['test_set']
    (X_gene_train, X_gene_test) = dataset['gene_set']
    n_classes = dataset['n_classes']
    
    train_labels = Y_train
    test_labels = Y_test
    n_classes = len(Y_train[1])
    
    ### DataLoader
    train_label = torch.as_tensor(train_labels)
    train = torch.tensor(X_gene_train.astype(np.float32))
    train_tensor = TensorDataset(train, train_label)
    train_loader = DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    test_label = torch.as_tensor(test_labels)
    test = torch.tensor(X_gene_test.astype(np.float32))
    test_tensor = TensorDataset(test, test_label)
    test_loader = DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=True)

    ### Hyperparameter
    softmax_hidden = 200 # 200
    dropout_rate = 0.2
    ensemble_model_num = 1
    n_gene = args.n_gene
    n_sm_out = n_classes
    n_embedding = args.n_embedding # 128
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    l2scale = args.l2scale
    fc_output_size = 64

    ### Model
    moBrca = moBRCAnet(
        data = X_gene_train,
        output_size = fc_output_size,
        n_features = n_gene,
        n_embedding = n_embedding,
        dropout_rate = dropout_rate
    ).to(device)

    if args.multi_omics == False:
        softmax_module = SoftmaxClassifier(
            n_embedding = 64,
            softmax_output = softmax_hidden,
            n_classes = n_classes,
            dropout_rate = dropout_rate
        ).to(device)

    if args.multi_omics == True:
        softmax_module = SoftmaxClassifier(
            n_embedding = 128,
            softmax_output = softmax_hidden,
            n_classes = n_classes,
            dropout_rate = dropout_rate
        ).to(device)

    print(moBrca)
    print(softmax_module)

    ### loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(moBrca.parameters(), lr=args.learning_rate)
    
    max_accr = 0
    max_pred = 0
    max_label = 0
    max_attn_gene = 0
    stop_point = 0
    
    # softmax_module.apply(init_weights)

    for epoch in range(args.epochs):
        
        sum_loss = 0
        sum_acc = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            x, y = x.to(device), y.to(device)
            if x.is_cuda != True:
                x = x.cuda()

            rep_gene, _ = moBrca(x)

            if args.multi_omics == False:
                outputs = softmax_module(rep_gene)

            if args.multi_omics == False:
                outputs = softmax_module(rep_gene)

            # print(f"output : {torch.argmax(outputs, dim=1)}")
            # print(f"output shape : {outputs.shape}")
            # print(f"y : {y}")
            # print(f"y shape : {y.shape}")
            loss = criterion(outputs, y)
            loss = torch.mean(loss)
            
            pred = torch.argmax(outputs, dim=1)
            label = torch.argmax(y, dim=1)
            correct_pred = torch.eq(pred, label)
            accuracy = torch.mean(correct_pred.float())
            
            sum_loss += loss
            sum_acc += accuracy
            
            loss.backward(retain_graph=True)
            optimizer.step()
        
        avg_loss = sum_loss / len(train_loader)
        avg_acc = sum_acc / len(train_loader)
        
        print("Epoch {:02d}/{:02d} Loss {:9.4f}, Accuracy {:9.4f}".format(
            epoch+1, args.epochs, avg_loss, avg_acc))
        wandb.log({
            "Loss": avg_loss,
            "Accuracy": avg_acc,
        })

        cur_acc = 0
        cur_pred = 0
        cur_label = 0
        cur_attn_gene = 0

        with torch.no_grad():
            for epoch in range(args.epochs):
                total_correct = 0
                total_samples = 0
                sum_loss = 0
                sum_acc = 0
                
                for batch_idx, (x, y) in enumerate(test_loader):
                    x, y = x.to(device), y.to(device)
                    if x.is_cuda != True:
                        x = x.cuda()

                    rep_gene, cur_attn_gene = moBrca(x)
                    outputs = softmax_module(rep_gene)

                    loss =  criterion(outputs, y)
                    cur_pred = torch.argmax(outputs, dim=1)
                    cur_label = torch.argmax(y, dim=1)
                    correct_pred = torch.eq(cur_pred, cur_label)
                    accuracy = torch.mean(correct_pred.float())

                sum_loss += loss
                avg_loss = sum_loss / len(test_loader)
                cur_acc = sum_acc / len(train_loader)
                # print(", cur_accr:%.6f," % cur_acc, "Train_batch_accr:%.6f, MAX:%.4f" % (cur_acc, max_accr), end='')

                wandb.log({
                    "Test Loss": avg_loss,
                    "Test Accuracy": cur_acc,
                })
                
        if stop_point > 10:
            break
        
        if max_accr > float(cur_acc):
            stop_point += 1
            
        if max_accr < float(cur_acc):
            max_accr = cur_acc
            max_pred = cur_pred
            max_label = cur_label
            max_attn_gene = cur_attn_gene
        print("")


    np.savetxt("./result/" + "prediction.csv", max_pred, fmt="%.0f", delimiter=",")
    np.savetxt("./result/" + "label.csv", max_label, fmt="%.0f", delimiter=",")
    np.savetxt("./result/" + "attn_score_gene.csv", max_attn_gene, fmt="%f", delimiter=",")
    print("ACCURACY : " + str(max_accr))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=136)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--l2scale",type=float, default=0.00001)
    parser.add_argument("--n_embedding",type=int, default=128)
    parser.add_argument("--fc_output",type=int, default=64)
    parser.add_argument("--n_gene",type=int, default=969)
    parser.add_argument("--multi_omics", type=lambda s: s. lower() in ['true', '1'], default=False)
    parser.add_argument("--train_x",type=str)
    parser.add_argument("--test_x",type=str)
    parser.add_argument("--train_y",type=str)
    parser.add_argument("--test_y",type=str)
    
    args = parser.parse_args()
    
    dataset = load_data(args.train_x, args.test_x, args.train_y, args.test_y, args.n_gene)
    main(args, dataset)
     
        

        
