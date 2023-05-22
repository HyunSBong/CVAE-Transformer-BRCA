import torch
import numpy as np
import pickle
import datetime
from datetime import datetime

import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

###########################
### Evaluation metrics  ###
###########################
def standardize(x, mean=None, std=None):
    """
    Shape x: (nb_samples, nb_vars)
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std


def upper_diag_list(m_):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    :param m_: numpy array of float. Shape=(N, N)
    :return: list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # Upper triangle of an array.
    tril = np.zeros_like(m_) + np.nan # m_과 비슷한 0으로 채워진 행렬 + NaN
    tril = np.tril(tril) # Lower triangle of an array.
    m += tril
    m = np.ravel(m) # 1 dimension으로 변환
    
    return m[~np.isnan(m)]

def pearson_correlation(x, y):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :return: Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        return (a - a_off) / a_std

    assert x.shape[0] == y.shape[0]
    
    x_ = standardize(x)
    y_ = standardize(y)
    
    return np.dot(x_.T, y_) / x.shape[0] # 내적

def correlations_list(x, y, corr_fn=pearson_correlation):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :param corr_fn: correlation function taking x and y as inputs
    """
    corr = corr_fn(x, y) # pearson_correlation
    return upper_diag_list(corr)

def gamma_coef(x, y):
    """
    Compute gamma coefficients for two given expression matrices
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z)
    """
    dists_x = 1 - correlations_list(x, x)
    dists_y = 1 - correlations_list(y, y)
    gamma_dx_dy = pearson_correlation(dists_x, dists_y)
    return gamma_dx_dy

def score_fn(x_test, x_gen):
    gamma_dx_dz = gamma_coef(x_test, x_gen)
    return gamma_dx_dz

###########################
###    UMAP metrics     ###
###########################
def tsne_2d(data, **kwargs):
    """
    Transform data to 2d tSNE representation
    :param data: expression data. Shape=(dim1, dim2)
    :param kwargs: tSNE kwargs
    :return:
    """
    print('... performing tSNE')
    tsne = TSNE(n_components=2, **kwargs)
    return tsne.fit_transform(data)


def plot_tsne_2d(data, labels, **kwargs):
    """
    Plots tSNE for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    dim1, dim2 = data.shape

    # Prepare label dict and color map
    label_set = set(labels)
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Perform tSNE
    if dim2 == 2:
        # print('plot_tsne_2d: Not performing tSNE. Shape of second dimension is 2')
        data_2d = data
    elif dim2 > 2:
        data_2d = tsne_2d(data, **kwargs)
    else:
        raise ValueError('Shape of second dimension is <2: {}'.format(dim2))

    # Plot scatterplot
    for k, v in label_dict.items():
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v)
    plt.legend()
    return plt.gca()

def scatter_2d(data_2d, labels, colors=None, **kwargs):
    """
    Scatterplot for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    # Prepare label dict and color map
    label_set = list(set(labels))[::-1]
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Plot scatterplot
    i = 0
    for k, v in label_dict.items():
        c = None
        if colors is not None:
            c = colors[i]
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v, color=c, **kwargs)
        i += 1
    plt.legend(markerscale=3,  fontsize=20)
    return plt.gca()

def scatter_2d_cancer(data_2d, labels, cancer, colors=None, **kwargs):
    # Prepare label dict and color map
    label_set = list(set(labels))[::-1]
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Plot scatterplot
    i = 0
    for k, v in label_dict.items():
        c = None
        if colors is not None:
            c = colors[i]

        idxs = np.logical_and(labels == v, cancer == 'normal')
        plt.scatter(data_2d[idxs, 0], data_2d[idxs, 1],
                    label=v, color=c, marker='o', s=7, **kwargs)
        idxs = np.logical_and(labels == v, cancer == 'cancer')
        plt.scatter(data_2d[idxs, 0], data_2d[idxs, 1], color=c, marker='+', **kwargs)
        i += 1
    lgnd = plt.legend(markerscale=3)
    return plt.gca()

def get_representation(tissue, datasets):
    cat_dicts = []

    tissues_dict_inv = np.array(list(sorted(set(tissue))))
    tissues_dict = {t: i for i, t in enumerate(tissues_dict_inv)}
    tissues = np.vectorize(lambda t: tissues_dict[t])(tissue)
    cat_dicts.append(tissues_dict_inv)

    dataset_dict_inv = np.array(list(sorted(set(datasets))))
    dataset_dict = {d: i for i, d in enumerate(dataset_dict_inv)}
    datasets = np.vectorize(lambda t: dataset_dict[t])(datasets)
    cat_dicts.append(dataset_dict_inv)

    cat_covs = np.concatenate((tissues[:, None], datasets[:, None]), axis=-1)
    cat_covs = np.int32(cat_covs)
    
    return dataset_dict_inv, cat_covs

def plot_umap(emb_2d, x_test, x_syn, test_tissue, test_dataset, syn_tissue, syn_dataset):
    
    dataset_dict_inv, cat_covs = get_representation(test_tissue, test_dataset)
    syn_dataset_dict_inv, syn_cat_covs = get_representation(syn_tissue, syn_dataset)
    
    x = np.concatenate((x_test, x_syn), axis=0)
    t = np.concatenate((test_tissue, syn_tissue), axis=0)
    s = np.array(['real'] * x_test.shape[0] + ['gen'] * x_syn.shape[0])
    c1 = np.array(['normal' if dataset_dict_inv[q] != 'tcga-t' else 'cancer' for q in cat_covs[:, 1]])
    c2 = np.array(['normal' if syn_dataset_dict_inv[q] != 'tcga-t' else 'cancer' for q in syn_cat_covs[:, 1]])
    c = np.concatenate((c1, c2), axis=0)
    print(f'x : {x.shape}')
    print(f't : {t.shape}')
    print(f'c : {c.shape}')
    print(f's : {s.shape}')

    # model = umap.UMAP(n_neighbors=300,
    #               min_dist=0.7,
    #               n_components=2,
    #               random_state=1111)
    # model.fit(x)
    # emb_2d = model.transform(x)

    plt.figure(figsize=(18, 6))
    
    ax = plt.gca()

    plt.subplot(1, 3, 1)
    colors =  plt.get_cmap('tab20').colors
    ax = scatter_2d(emb_2d, t, colors=colors, s=12, marker='.')
    ax = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=3, markerscale=5, fontsize=15)
    # plt.legend(fontsize=10)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    colors = ['brown', 'lightgray']
    # colors = ['lightgray', 'brown']
    ax = scatter_2d(emb_2d, c, colors=colors, s=12, marker='.')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3, markerscale=5, fontsize=15)
    # plt.legend(fontsize=100)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # colors = ['lightgray', 'blue']
    # colors = ['lightgray', 'skyblue']
    colors = ['blue', 'lightgray']
    ax = scatter_2d(emb_2d, s, colors=colors, s=12, marker='.')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3, markerscale=5, fontsize=15)
    # plt.legend(fontsize=20)
    plt.axis('off')

###########################
### generate_synthetic  ###
###########################
def generate_synthetic(vae_model, le, X, gene_names, Y_test_tissues):
    genes_to_validate = 40
    original_means = np.mean(X, axis=0)
    original_vars = np.var(X, axis=0)

    with torch.no_grad():
        all_samples = Y_test_tissues
        
        le = LabelEncoder()
        onehot_c = le.fit_transform(all_samples)
        
        c = all_samples # ['bladder' 'bladder' 'bladder' ... 'uterus' 'uterus' 'uterus']
        c = torch.from_numpy(onehot_c) # [0 0 0 ... 14 14 14]
        x = vae_model.inference(n=len(all_samples), c=c)
        print(f'generated x => {x}')

    sampled_means = np.mean(x.detach().cpu().numpy(), axis=0)
    sampled_vars = np.var(x.detach().cpu().numpy(), axis=0)

    x_syn = x.detach().cpu().numpy() # (7500,1000)
    print(f'x_syn.shape : {x_syn.shape}')
    return x_syn

def get_representation(tissue, datasets):
    cat_dicts = []

    tissues_dict_inv = np.array(list(sorted(set(tissue))))
    tissues_dict = {t: i for i, t in enumerate(tissues_dict_inv)}
    tissues = np.vectorize(lambda t: tissues_dict[t])(tissue)
    cat_dicts.append(tissues_dict_inv)

    dataset_dict_inv = np.array(list(sorted(set(datasets))))
    dataset_dict = {d: i for i, d in enumerate(dataset_dict_inv)}
    datasets = np.vectorize(lambda t: dataset_dict[t])(datasets)
    cat_dicts.append(dataset_dict_inv)

    cat_covs = np.concatenate((tissues[:, None], datasets[:, None]), axis=-1)
    cat_covs = np.int32(cat_covs)
    
    return dataset_dict_inv, cat_covs

###########################
###    save_synthetic   ###
###########################
def save_synthetic(vae_model, x, y_test, epoch, batch_size, lr, dim_size):
    model_dir = '../checkpoints/models/cvae/'

    with torch.no_grad():
        x_syn = x.detach().cpu().numpy() # (7500,1000)
        
        date_val = datetime.today().strftime("%Y%m%d%H%M")
        
        file = f'../../checkpoints/models/cvae/gen_brca_cvae_{date_val}_bat{batch_size}_epoch{epoch}_dim{dim_size}_lr{lr}_.pkl'
        data = {'model': vae_model,
                'x_syn': x_syn,
                'y_syn': y_test,
                }
        with open(file, 'wb') as files:
            pickle.dump(data, files)
            
    return x_syn

def generate_synthetic_n_save(vae_model, le, X, gene_names, Y_test_tissues, epoch, trial_name, dim_size):
    genes_to_validate = 40
    original_means = np.mean(X, axis=0)
    original_vars = np.var(X, axis=0)
    model_dir = '../checkpoints/models/cvae/'

    with torch.no_grad():
        # number_of_samples = 500
        # labels_to_generate = []
        x_synthetic = []
        y_synthetic = []
        
        # for label_value in label_encoder.classes_:
        #     label_to_generate = [label_value for i in range(samples_per_labels)]
        #     labels_to_generate += label_to_generate
        # all_samples = np.array(labels_to_generate)
        all_samples = Y_test_tissues
        le = LabelEncoder()
        onehot_c = le.fit_transform(all_samples)
        
        c = all_samples # ['bladder' 'bladder' 'bladder' ... 'uterus' 'uterus' 'uterus']
        c = torch.from_numpy(onehot_c) # [0 0 0 ... 14 14 14]
        x = vae_model.inference(n=len(all_samples), c=c)
        
        x_syn = x.detach().cpu().numpy() # (7500,1000)

        x_synthetic += list(x.detach().cpu().numpy())
        y_synthetic += list(np.ravel(le.transform(all_samples)))
        
        print(f'x_syn.shape : {x_syn.shape}')
        date_val = datetime.today().strftime("%Y%m%d%H%M")
        
        # pd.DataFrame(x_synthetic, columns=gene_names).to_csv(f'{model_dir}gen_rnaseqdb_cvae_{date_val}_{trial_name}_epoch{epoch}_dim{dim_size}_expressions.csv', index=False)
        # pd.DataFrame(y_synthetic, columns=['label']).to_csv(f'{model_dir}gen_rnaseqdb_cvae_{date_val}_{trial_name}_epoch{epoch}_dim{dim_size}_labels.csv', index=False)
   
        file = f'../checkpoints/models/cvae/gen_rnaseqdb_cvae_{date_val}_{trial_name}_epoch{epoch}_dim{dim_size}_.pkl'
        data = {'model': vae_model,
                'x_syn': x_syn
                }
        with open(file, 'wb') as files:
            pickle.dump(data, files)
            
    return x_syn