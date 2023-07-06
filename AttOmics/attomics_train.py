from torch.utils.data import Dataset
from pathlib import Path
import sys
import os 
import numpy as np
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

path_root = Path(os.path.abspath(''))
# path_root
sys.path.append(str(path_root))

from AttOmics import AttOmics
import torch


class OmicsDataset(Dataset):
    def __init__(self, omics, label, event=None):
            self.omics = omics
            self.label = label
            self.event = event
    
    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        sample = {"x": self.omics[index], "label": self.label[index]}
        if self.event is not None:
            sample.update({"event": self.event[index]})
        
        sample = {k: torch.as_tensor(v) for k,v in sample.items()}
        return sample
    
def main(args):
    train = pd.read_csv(args.train_x, sep = '\t')
    X = train.iloc[:,1:-1].values
    Y = train.iloc[:,-1].values
    val = pd.read_csv(args.val_x, sep = '\t')
    X_val = val.iloc[:,1:-1].values
    Y_val = val.iloc[:,-1].values

    # create dataset 
    dataset = OmicsDataset(omics=X, label=Y)
    # create dataloader
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # create dataset 
    dataset_val = OmicsDataset(omics=X_val, label=Y_val)
    # create dataloader
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, drop_last=False)
    
    n_class = np.unique(Y).shape[0]
    label_count = np.bincount(Y)
    class_weights = (label_count.max() / label_count).astype(np.float32)
    
    optimizer={"class_path": "torch.optim.Adam", "init_args": {"lr": 0.0001}}
    lr_scheduler={"class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau", 
      "init_args":
        {"mode": "min",
        "factor": 0.1,
        "patience": 10,
        "threshold": 0.0001}}

    model = AttOmics(
        n_group= 10,
        n_layers= 1,
        num_heads= 1,
        attention_norm= "layer_norm",
        grouping_method= "random",
        head_norm= "layer_norm",
        sa_residual_connection= True,
        head_residual_connection= False,
        head_dropout= 0.0,
        head_batch_norm= False,
        reuse_grp= True,
        constant_group_size= False,
        head_input_dim= 500,
        head_hidden_ratio=[0.5],
        input_dim=X.shape[1],  # a dict of dimension
        num_classes=n_class,
        label_type="cancer_type",
        class_weights=class_weights,
        train_data=X,
        optimizer_init=optimizer,
        scheduler_init=lr_scheduler)
    
    model.label_str = ['Normal', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-COAD',
           'TCGA-HNSC', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG',
           'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-OV', 'TCGA-PRAD',
           'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-THCA', 'TCGA-UCEC']
    
    trainer = Trainer(gpus=[args.gpu_id], 
                  logger=MLFlowLogger(experiment_name="AttOmics",save_dir= "./logs"),
                  max_epochs=50,
                  )
    
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_x",type=str)
    parser.add_argument("--val_x",type=str)
    
    args = parser.parse_args()
    main(args)