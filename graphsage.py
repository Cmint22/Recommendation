import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import os

# ------------------------- GraphSAGE Model ------------------------- #
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout=0.2, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.activation = getattr(F, activation)
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

# ------------------------- Utility Functions ------------------------- #
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep=' ', names=["user", "item", "rating"])
    test_df = pd.read_csv(test_path, sep=' ', names=["user", "item", "rating"])
    num_users = max(train_df['user'].max(), test_df['user'].max()) + 1
    num_items = max(train_df['item'].max(), test_df['item'].max()) + 1

    user_tensor = torch.tensor(train_df['user'].values, dtype=torch.long)
    item_tensor = torch.tensor(train_df['item'].values, dtype=torch.long)
    edge_index = torch.stack([torch.cat([user_tensor, item_tensor + num_users]),
                              torch.cat([item_tensor + num_users, user_tensor])])

    x = torch.randn(num_users + num_items, 64)  # random initial features
    data = Data(x=x, edge_index=edge_index)
    return data, train_df, test_df, num_users, num_items

def get_user_positive_items(train_df):
    pos_items = defaultdict(set)
    for user, item in zip(train_df['user'], train_df['item']):
        pos_items[user].add(item)
    return pos_items

def evaluate(user_emb, item_emb, test_df, train_pos, k_list=[10, 20, 30, 50]):
    all_metrics = {k: {"HR": 0, "P": 0, "R": 0, "NDCG": 0} for k in k_list}
    user_item_matrix = torch.matmul(user_emb, item_emb.t())

    for user in test_df['user'].unique():
        known_pos = train_pos[user]
        test_items = set(test_df[test_df['user'] == user]['item'])
        scores_user = user_item_matrix[user].detach().cpu().numpy()
        scores_user[list(known_pos)] = -np.inf
        top_items_idx = np.argsort(-scores_user)[:max(k_list)]

        for k in k_list:
            top_k = top_items_idx[:k]
            hits = sum([1 for item in top_k if item in test_items])
            precision = hits / k
            recall = hits / len(test_items) if len(test_items) > 0 else 0
            ndcg = sum([1 / np.log2(idx + 2) if top_items_idx[idx] in test_items else 0 for idx in range(k)])
            all_metrics[k]["HR"] += (hits > 0)
            all_metrics[k]["P"] += precision
            all_metrics[k]["R"] += recall
            all_metrics[k]["NDCG"] += ndcg

    num_users = len(test_df['user'].unique())
    for k in k_list:
        for m in all_metrics[k]:
            all_metrics[k][m] /= num_users
    return all_metrics

# ------------------------- Training Function ------------------------- #
def train_model(config):
    data, train_df, test_df, num_users, num_items = load_data("./data/train.txt", "./data/test.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(in_channels=64, 
                      hidden_channels=config['hidden_channels'], 
                      out_channels=64, 
                      n_layers=config['n_layers'], 
                      dropout=config['dropout'], 
                      activation=config['activation']).to(device)
    data = data.to(device)

    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        user_emb = out[:num_users]
        item_emb = out[num_users:]

        pos_u = torch.tensor(train_df['user'].values, dtype=torch.long).to(device)
        pos_i = torch.tensor(train_df['item'].values, dtype=torch.long).to(device)
        neg_i = torch.randint(0, num_items, (len(pos_u),)).to(device)

        user_vec = user_emb[pos_u]
        pos_vec = item_emb[pos_i]
        neg_vec = item_emb[neg_i]

        if config['loss_type'] == 'bpr':
            pos_scores = (user_vec * pos_vec).sum(dim=-1)
            neg_scores = (user_vec * neg_vec).sum(dim=-1)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        elif config['loss_type'] == 'bce':
            logits = torch.matmul(user_vec, item_emb.t())
            labels = torch.zeros_like(logits)
            labels[range(len(pos_u)), pos_i] = 1.0
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            raise ValueError("Invalid loss_type")

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        user_emb = out[:num_users]
        item_emb = out[num_users:]
        train_pos = get_user_positive_items(train_df)
        return evaluate(user_emb, item_emb, test_df, train_pos)

# ------------------------- Hyperparameter Tuning ------------------------- #
def tune_hyperparameters():
    default_config = {
        'hidden_channels': 64,
        'n_layers': 2,
        'dropout': 0.2,
        'activation': 'relu',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'optimizer': 'Adam',
        'loss_type': 'bpr'
    }

    param_grid = {
        'hidden_channels': [32, 64, 128],
        'n_layers': [1, 2, 3],
        'dropout': [0.0, 0.2, 0.5],
        'activation': ['relu', 'gelu'],
        'lr': [0.001, 0.01, 0.05],
        'weight_decay': [0.0, 1e-4, 1e-3],
        'optimizer': ['Adam', 'AdamW', 'SGD'],
        'loss_type': ['bpr', 'bce']
    }

    for param_name, values in param_grid.items():
        print(f"Tuning {param_name}...")
        for val in values:
            config = default_config.copy()
            config[param_name] = val
            print(f"  → {param_name} = {val}")
            metrics = train_model(config)
            for k in [10, 20, 30, 50]:
                print(f"    Recall@{k}: {metrics[k]['R']:.4f}  Precision@{k}: {metrics[k]['P']:.4f}  HR@{k}: {metrics[k]['HR']:.4f}  NDCG@{k}: {metrics[k]['NDCG']:.4f}")
            print("\n")

if __name__ == "__main__":
    tune_hyperparameters()