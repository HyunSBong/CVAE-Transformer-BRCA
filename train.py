import os
import numpy as np
import time
import argparse
import pandas as pd
import pickle
import random
import datetime
from datetime import datetime
from collections import defaultdict
from utils import *
from model import CVAE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import wandb

def load_data(num_genes, multivariate, train_path, test_path, val_path):
    df_real = pd.read_csv(f'./data/tcga_brca.csv')
    
    df_real_gene_columns = df_real.iloc[:,2:-3].columns
    train_genes = list(df_real_gene_columns)
    gene_names = train_genes

    data_train = np.load(train_path, allow_pickle=True)
    data_val = np.load(val_path, allow_pickle=True)
    data_test = np.load(test_path, allow_pickle=True)

    data = {'train_set': (data_train['x'], data_train['y']),
            'test_set': (data_test['x'], data_test['y']),
            'val_set': (data_val['x'], data_val['y']),
            'gene_names': gene_names
           }
    return data

def main(args, rna_dataset):
    # wandb
    wandb.init(project="TGCA BRCA init", reinit=True)
    wandb.config.update(args)
    
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # GPU 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # use bernoulli decoder
    multivariate = False
    if args.multivariate == 1:
        # use gaussian decoder
        multivariate = True
    
    # Train/Test dataset & tissues
    # X = rna_dataset['X'] # [COL4A1, IFT27,,,].values
    # Y = rna_dataset['Y'] # [["PAM50"]].values
    (X_train, Y_train) = rna_dataset['train_set']
    (X_test, Y_test) = rna_dataset['test_set']
    (X_val, Y_val) = rna_dataset['val_set']
    
    # Standardization & Normalization data
    std_scaler = StandardScaler().fit(X_train)
    X_train = std_scaler.transform(X_train)
    X_test = std_scaler.transform(X_test)
    
    X = X_train
    
    gene_names = rna_dataset['gene_names'] # ex) COL4A1, IFT27,,,
    
    num_tissue = 5 #len(set(Y)) # 15 (breast, lung, liver 등 15개 tissue)
    view_size = X.shape[1]
    
    Y_train_tissue_datasets = Y_train
    Y_test_tissue_datasets = Y_test
    Y_val_tissue_datasets = Y_val
    
    # one-hot encoding
    le = LabelEncoder()
    le.fit(Y_train_tissue_datasets)
    train_labels = le.transform(Y_train_tissue_datasets) # bladder,uterus,,, -> 0,14,,,
    test_labels = le.transform(Y_test_tissue_datasets)
    val_labels = le.transform(Y_val_tissue_datasets)
    
    # DataLoader
    train_label = torch.as_tensor(train_labels)
    train = torch.tensor(X_train.astype(np.float32))
    train_tensor = TensorDataset(train, train_label)
    train_loader = DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    test_label = torch.as_tensor(test_labels)
    test = torch.tensor(X_test.astype(np.float32))
    test_tensor = TensorDataset(test, test_label)
    test_loader = DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=True)
    
    val_label = torch.as_tensor(val_labels)
    val = torch.tensor(X_val.astype(np.float32))
    val_tensor = TensorDataset(val, val_label)
    val_loader = DataLoader(dataset=val_tensor, batch_size=args.batch_size, shuffle=True)
    
    # loss function
    mse_criterion = nn.MSELoss(size_average=False, reduction="sum")
    def loss_fn_gaussian(x, mean, log_var, z_mean, z_sigma, beta):
        # reconstruction error
        reconstr_loss = mse_criterion(
            z_mean, x)
            
        # Kullback-Leibler divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # kl_loss *= beta
        
        elbo = (reconstr_loss + kl_loss) / x.size(0)
        
        return {'elbo': elbo, 'reconstr_loss': reconstr_loss, 'kl_loss': kl_loss}
    
    def loss_fn_bernoulli(recon_x, x, mean, log_var):
        # reconstruction error
        reconstr_loss = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, view_size), x.view(-1, view_size), reduction='sum')
        # Kullback-Leibler divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
        elbo = (reconstr_loss + kl_loss) / x.size(0)
        
        return {'elbo': elbo, 'reconstr_loss': reconstr_loss, 'kl_loss': kl_loss}
    
    compress_dims = [969, 512, 256]
    decompress_dims = [256, 512, 969]
    hidden_dims = args.hidden_dims
    if hidden_dims == 2:
        compress_dims = [512, 256]
        decompress_dims = [256, 512]
    elif hidden_dims == 3:
        compress_dims = [969, 512, 256]
        decompress_dims = [256, 512, 969]
    elif hidden_dims == 4:
        compress_dims = [969, 512, 256, 128]
        decompress_dims = [128, 256, 512, 969]
    
    print(compress_dims)
    print(decompress_dims)
    vae = CVAE(
        data_dim=X_train.shape[1],
        compress_dims=compress_dims,
        latent_size=args.latent_size,
        decompress_dims=decompress_dims,
        conditional=args.conditional,
        view_size = view_size,
        multivariate = multivariate,
        num_labels=num_tissue if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    wandb.watch(vae)
    
    # Train
    stop_point = 10
    best_score = 0.0000000000001
    initial_stop_point = stop_point
    stop_point_done = False
    losses = []
    
    score = 0
    beta = args.beta

    for epoch in range(args.epochs):
        train_loss = 0
        sum_elbo = 0
        sum_kl_loss = 0
        sum_reconstr_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if x.is_cuda != True:
                x = x.cuda()

            if args.conditional and multivariate:
                mean, log_var, z_mean, z_sigma = vae(x, y)
                losses = loss_fn_gaussian(x, mean, log_var, z_mean, z_sigma, beta)
            elif args.conditional and multivariate==False:
                recon_x, mean, log_var, z = vae(x, y)
                losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
            else:
                recon_x, mean, log_var, z = vae(x)
                losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
            
            loss = losses['elbo'].clone() #  KL-Divergence + reconstruction error / x.size(0)
            train_loss += loss
            
            sum_elbo += losses['elbo']
            sum_kl_loss += losses['kl_loss']
            sum_reconstr_loss += losses['reconstr_loss']
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    for epoch_v in range(args.epochs):
                        test_loss = 0
                        sum_elbo = 0
                        sum_kl_loss = 0
                        sum_reconstr_loss = 0

                        for batch_idx, (x, y) in enumerate(val_loader):
                            x, y = x.to(device), y.to(device)
                            if x.is_cuda != True:
                                x = x.cuda()

                            if args.conditional and multivariate:
                                mean, log_var, z_mean, z_sigma = vae(x, y)
                                losses = loss_fn_gaussian(x, mean, log_var, z_mean, z_sigma, beta)
                            elif args.conditional and multivariate==False:
                                recon_x, mean, log_var, z = vae(x, y)
                                losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
                            else:
                                recon_x, mean, log_var, z = vae(x)
                                losses = loss_fn_bernoulli(recon_x, x, mean, log_var)

                            loss = losses['elbo'].clone() #  KL-Divergence + reconstruction error / x.size(0)
                            test_loss += loss

                            sum_elbo += losses['elbo']
                            sum_kl_loss += losses['kl_loss']
                            sum_reconstr_loss += losses['reconstr_loss']

                        avg_val_loss = sum_elbo / len(val_loader)
                        avg_kl_loss = sum_kl_loss / len(val_loader)
                        avg_reconstr_loss = sum_reconstr_loss / len(val_loader)

                        # print("Epoch {:02d}/{:02d} Test Loss {:9.4f}, KL {:9.4f}, Reconstruction {:9.4f}".format(
                        #     epoch, args.epochs, avg_val_loss, avg_kl_loss, avg_reconstr_loss))

                        wandb.log({
                            "ELBO Validation Loss": avg_val_loss,
                            # "Reconstruction Error": avg_reconstr_loss,
                            # "KL-Divergence": avg_kl_loss
                        })

        print(f'stop point : {stop_point}')
        c = torch.from_numpy(test_labels) # le.fit_transform(Y_train_tissues)
        x_syn = vae.inference(n=c.size(0), c=c)
        score = score_fn(X_test, x_syn.detach().cpu().numpy())
        if score > best_score or epoch % 50 == 0:
            best_score = score
            stop_point = initial_stop_point
            x_syn = save_synthetic(vae, x_syn, Y_test, epoch+1, args.batch_size, args.learning_rate, X.shape[1], best_score)
        else:
            stop_point -= 1
        
        avg_loss = sum_elbo / len(train_loader)
        avg_kl_loss = sum_kl_loss / len(train_loader)
        avg_reconstr_loss = sum_reconstr_loss / len(train_loader)
        
        print(f'beta : {beta}')
        print("Epoch {:02d}/{:02d} Loss {:9.4f}, KL {:9.4f}, Reconstruction {:9.4f}".format(
            epoch+1, args.epochs, avg_loss, avg_kl_loss, avg_reconstr_loss))
        print(f'==>Gamma Score : {score}')
        wandb.log({
            "ELBO Loss": avg_loss,
            "Reconstruction Error": avg_reconstr_loss,
            "KL-Divergence": avg_kl_loss,
            "Gamma_Score": score,
            # "beta": beta
        })
        
            

    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            sum_elbo = 0
            sum_kl_loss = 0
            sum_reconstr_loss = 0

            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                if x.is_cuda != True:
                    x = x.cuda()

                if args.conditional and multivariate:
                    mean, log_var, z_mean, z_sigma = vae(x, y)
                    losses = loss_fn_gaussian(x, mean, log_var, z_mean, z_sigma, beta)
                elif args.conditional and multivariate==False:
                    recon_x, mean, log_var, z = vae(x, y)
                    losses = loss_fn_bernoulli(recon_x, x, mean, log_var)
                else:
                    recon_x, mean, log_var, z = vae(x)
                    losses = loss_fn_bernoulli(recon_x, x, mean, log_var)

                loss = losses['elbo'].clone() #  KL-Divergence + reconstruction error / x.size(0)
                test_loss += loss

                sum_elbo += losses['elbo']
                sum_kl_loss += losses['kl_loss']
                sum_reconstr_loss += losses['reconstr_loss']

            avg_val_loss = sum_elbo / len(test_loader)
            avg_kl_loss = sum_kl_loss / len(test_loader)
            avg_reconstr_loss = sum_reconstr_loss / len(test_loader)

            print("Epoch {:02d}/{:02d} Test Loss {:9.4f}, KL {:9.4f}, Reconstruction {:9.4f}".format(
                epoch, args.epochs, avg_val_loss, avg_kl_loss, avg_reconstr_loss))

            wandb.log({
                "ELBO Test Loss": avg_val_loss,
                # "Reconstruction Error": avg_reconstr_loss,
                # "KL-Divergence": avg_kl_loss
            })

    x_syn = save_synthetic(vae, x, Y_test, args.epochs, args.batch_size, args.learning_rate, X.shape[1])
    # draw_umap(X_test, x_syn, Y_test_tissues, Y_test_datasets)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1300)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-04) # bernoulli 0.001
    parser.add_argument("--l2scale",type=float, default=0.00001)
    parser.add_argument("--compress_dims", type=list, default=[1000, 512, 256])
    parser.add_argument("--decompress_dims", type=list, default=[256, 512, 1000])
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--num_genes", type=int, default=18154)
    parser.add_argument("--multivariate", type=int, default=1)
    parser.add_argument("--geo", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1) # 0.144
    parser.add_argument("--hidden_dims", type=int, default=3) # 0.144
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--val_path", type=str)

    args = parser.parse_args()
    
    rna_dataset = load_data(args.num_genes, args.multivariate, args.train_path, args.test_path, args.val_path)

    main(args, rna_dataset)
