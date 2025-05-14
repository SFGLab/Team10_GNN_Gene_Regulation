from torch_geometric.data import Data
from torch_geometric.utils import convert
import torch
import pandas as pd
import torch_geometric.transforms as T
import tensorflow as tf
import itertools
import numpy as np
import torch
from torch.nn import Linear
import seaborn as sns
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import HeteroData

from torch_geometric.utils import negative_sampling

from torch_geometric.nn import ChebConv
from torch_geometric.nn import HypergraphConv, GATConv
import torch.nn.functional as F

#FOR VISUALIZING GRAPH
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import Counter
from torch_geometric.utils import to_networkx

import os
from clearml import Task

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import DetCurveDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    matthews_corrcoef, jaccard_score, cohen_kappa_score,
    f1_score, top_k_accuracy_score, auc
)

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    matthews_corrcoef, jaccard_score, cohen_kappa_score,
    f1_score, top_k_accuracy_score, auc
)

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden1_channels, hidden2_channels, out_channels, dec, af_val, num_layers, epoch, aggr, var):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        if(var=="ChebConv"):
            self.convs.append(eval(var)(in_channels, hidden1_channels, aggr=aggr, K=3))
            for _ in range(num_layers - 2):
                self.convs.append(eval(var)(hidden1_channels, hidden1_channels, aggr=aggr, K=3))
            self.convs.append(eval(var)(hidden1_channels, out_channels, aggr=aggr, K=3))
        else:
            self.convs.append(eval(var)(in_channels, hidden1_channels, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(eval(var)(hidden1_channels, hidden1_channels, aggr=aggr))
            self.convs.append(eval(var)(hidden1_channels, out_channels, aggr=aggr))

    def encode(self, x, edge_index, af_val):
        prev_x = None
        for i, conv in enumerate(self.convs[:-1]):
            prev_x = x
            x = conv(x, edge_index)
            if i > 0:
                x = x + prev_x
            x = eval(af_val)(x)
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index, dec):
        if(dec=="dot_sum"):
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        else:
            cos = torch.nn.CosineSimilarity(dim=1)
            output = cos(z[edge_label_index[0]].float(), z[edge_label_index[1]].float())
            return output


def train_link_predictor(model, train_data, val_data, optimizer, scheduler, criterion, n_epochs, af_val, dec, model_id):
    logger = Task.current_task().get_logger()

    # Pre-sample negative edges for validation to ensure consistency
    val_neg_edge_index = negative_sampling(
        edge_index=val_data.edge_index,
        num_nodes=val_data.num_nodes,
        method='sparse'
    )

    # Ensure balanced validation set with equal positive and negative samples
    num_val_pos = val_data.edge_label_index.size(1) // 2  # Half of the validation edges are positive
    val_pos_edge_index = val_data.edge_label_index[:, :num_val_pos]

    # Create balanced validation set
    val_edge_label_index = torch.cat([val_pos_edge_index, val_neg_edge_index], dim=-1)
    val_edge_label = torch.cat([
        torch.ones(val_pos_edge_index.size(1), device=val_data.x.device),
        torch.zeros(val_neg_edge_index.size(1), device=val_data.x.device)
    ])

    # Store these for evaluation
    val_data.edge_label_index_balanced = val_edge_label_index
    val_data.edge_label_balanced = val_edge_label

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index, af_val)

        # Sample negative edges for training
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1),
            method='sparse'
        )

        # Create balanced training set
        edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            train_data.edge_label,
            torch.zeros(neg_edge_index.size(1), device=train_data.x.device)
        ])

        out = model.decode(z, edge_label_index, dec).view(-1)
        loss = criterion(out, edge_label.float())
        loss.backward()

        # Logging training loss
        logger.report_scalar("loss", "train", iteration=epoch, value=loss.item())

        # Evaluation with balanced validation set
        val_auc, precision, recall, fpr, tpr, mcc, jac_score, cohkap_score, f1, top_k, val_loss = eval_link_predictor(
            model, val_data, af_val, dec, criterion, epoch, use_balanced=True
        )
        print("Epoch: ", epoch, "Loss: ", loss.item(), "Val Loss: ", val_loss, "f1: ", f1, 'AUC: ', val_auc)
        scheduler.step(val_loss)
        optimizer.step()
        # Logging validation metrics
        logger.report_scalar("AUC", "val", iteration=epoch, value=val_auc)
        logger.report_scalar("F1 Score", "val", iteration=epoch, value=f1)
        logger.report_scalar("MCC", "val", iteration=epoch, value=mcc)
        logger.report_scalar("Jaccard", "val", iteration=epoch, value=jac_score)
        logger.report_scalar("Cohen Kappa", "val", iteration=epoch, value=cohkap_score)
        logger.report_scalar("Top-K Accuracy", "val", iteration=epoch, value=top_k)

    return model


@torch.no_grad()
def eval_link_predictor(model, data, af_val, dec, criterion=None, epoch=None, use_balanced=False):
    model.eval()
    logger = Task.current_task().get_logger()
    z = model.encode(data.x, data.edge_index, af_val)

    # Use the balanced validation set if available and requested
    if use_balanced and hasattr(data, 'edge_label_index_balanced'):
        edge_label_index = data.edge_label_index_balanced
        edge_label = data.edge_label_balanced
    else:
        # For test data or if balanced not available, create balanced set on the fly
        pos_edge_index = data.edge_label_index
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
            method='sparse'
        )
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            data.edge_label,
            torch.zeros(neg_edge_index.size(1), device=data.x.device)
        ])

    out = model.decode(z, edge_label_index, dec)

    loss = criterion(out.view(-1), edge_label.float())
    logger.report_scalar("loss", "val", iteration=epoch or 0, value=loss.item())
    out = torch.sigmoid(out)
    print(out, edge_label)
    # Convert to CPU for sklearn metrics
    actual = edge_label.cpu().numpy()
    pred_scores = out.cpu().numpy()

    # Calculate metrics
    auc = roc_auc_score(actual, pred_scores)
    fpr, tpr, _ = roc_curve(actual, pred_scores)
    precision, recall, _ = precision_recall_curve(actual, pred_scores)

    # Binary predictions
    pred_binary = pred_scores.copy()
    pred_binary[pred_binary > 0.5] = 1
    pred_binary[pred_binary <= 0.5] = 0

    # Classification metrics
    mcc = matthews_corrcoef(actual, pred_binary)
    jac_score = jaccard_score(actual, pred_binary)
    cohkap_score = cohen_kappa_score(actual, pred_binary)
    f1 = f1_score(actual, pred_binary)
    top_k = top_k_accuracy_score(actual, pred_scores, k=1)

    return auc, precision, recall, fpr, tpr, mcc, jac_score, cohkap_score, f1, top_k, loss


def main_run(hidden1_channels, hidden2_channels, out_channels, data, dec, af_val, num_layers, epoch, aggr, var, device):
    auprs = 0
    aucs = 0

    # Multiple runs for stability
    for i in range(1):
        # Use fixed seed for reproducibility
        torch.manual_seed(42 + i)

        # Create data split
        split = RandomLinkSplit(
            num_val=0.1,
            num_test=0.2,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
        )

        train_data, val_data, test_data = split(data)

        # Initialize model
        in_channels = data.num_features
        model = Net(
            in_channels, hidden1_channels, hidden2_channels, out_channels,
            dec, af_val, num_layers, epoch, aggr, var
        ).to(device)

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        criterion = torch.nn.BCEWithLogitsLoss()
        task.set_parameters({
                "model_architecture": model.__class__.__name__,
                "decoder": dec,
                "activation_function": af_val,
                "num_layers": num_layers,
                "optimizer": optimizer.__class__.__name__,
                "initial_learning_rate": optimizer.param_groups[0]['lr'],
                "epochs": epoch,
                #"patience": patience,
                #"weight_decay_factor": weight_decay_factor,
                #"min_learning_rate": min_lr,
                #"monitor_metric": monitor_metric
            })
        # Train model
        model = train_link_predictor(
            model, train_data, val_data, optimizer, scheduler, criterion, epoch, af_val, dec, 'test_basic'
        ).to(device)

        # Evaluate on test set (with balanced negative sampling)
        test_auc, precision, recall, fpr, tpr, mcc, jac_score, cohkap_score, f1, top_k, test_loss = eval_link_predictor(
            model, test_data, af_val, dec, criterion, use_balanced=True
        )

        # Calculate metrics
        aucs += test_auc
        aupr = auc(recall, precision)
        auprs += aupr

    # Average metrics over all runs
    mean_auc = float(aucs/10)
    mean_aupr = float(auprs/10)

    # Create results dataframe
    result = {
        #"org": org,
        #"ds": ds,
        "dec": dec,
        "af_val": af_val,
        "num_layers": num_layers,
        "epoch": epoch,
        "aggr": aggr,
        "var": var,
        "auc": mean_auc,
        "aupr": mean_aupr,
        "mcc": mcc,
        "jac_score": jac_score,
        "cohkap_score": cohkap_score,
        "f1": f1,
        "top_k": top_k
    }

    df = pd.DataFrame([result])
    return df, model


if __name__=='main':

    path = "/content/drive/MyDrive/BEELINE"
    gold_std = "/content/drive/MyDrive/BEELINE/basic_data_aug_hESC.pt"


    parameters = {
        "decoder": ["dot_sum"],# substitute "cos" to use cos decoder
        "af": ["F.silu"],#"F.sigmoid"],  # substitute "F.silu","F.tanh" for activation functions
        "num_layers" : [3],#2,4,5 #substitute layer count
        "variant" : ["HypergraphConv"],#,"ChebConv", 'GATConv'],#,"SSGConv","ChebConv","ClusterGCNConv" #substitute convolution layers
        "aggrs" :["sum"],# substitute "add"
        "epochs" :[150]# substitute epochs 100,150,200,250
                }



    hidden1_channels=128
    hidden2_channels=64
    out_channels= 32
    final_out = []

    #ds_type =["basic_aug_data_"] # Try different graph types : "basic_TS_data_","basic_TS_aug_data_","basic_data_"

    # final_out = pd.DataFrame()
    data = torch.load(gold_std, weights_only=False)#+ds+org+"_"+files[0]+".pt")#.cuda()
    data.x = data.x.to(torch.float32)
    data.edge_index = data.edge_index.to(torch.int64)
    for dec in parameters.get('decoder'):
        for lay_num in parameters.get('num_layers'):
            for epoch in parameters.get('epochs'):
                for af_val in parameters.get('af'):
                    for aggr in parameters.get('aggrs'):
                        for var in parameters.get('variant'):
                            task = Task.create(
                                project_name="GNN-GRN",
                                task_name=var+str(lay_num)+"_"+dec+"_scheduler",
                                task_type=Task.TaskTypes.training  # or .inference, .data_processing, etc.
                            )

                            temp,model = main_run(hidden1_channels,hidden2_channels,out_channels,data,dec,af_val,lay_num,epoch,aggr,var, device)
                            final_out.append(temp)
                            task.close()

    print(pd.concat(final_out))
    # final_out.to_excel('/content/drive/MyDrive/Colab Notebooks/DREAM challenge GCN GNN Experimentation/DREAM5_InsilicoSize100_'+str(org)+'_'+var+'Dimension_count.xlsx', index=False)