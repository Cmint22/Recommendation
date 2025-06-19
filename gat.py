import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import os

# ----------------------------- GAT Model ----------------------------- #
class GATRec(nn.Module):
    def __init__(self, num_users, num_items, in_channels, hidden_channels, out_channels,
                 num_heads, dropout, edge_dropout, neg_slope):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, in_channels)
        self.item_embedding = nn.Embedding(num_items, in_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads,
                            dropout=edge_dropout, negative_slope=neg_slope)
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1,
                            dropout=edge_dropout, negative_slope=neg_slope)
        self.dropout = dropout
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        user_emb = x[:self.user_embedding.num_embeddings]
        item_emb = x[self.user_embedding.num_embeddings:]
        return user_emb, item_emb

# --------------------------- Data Utilities --------------------------- #
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep=' ', names=["user", "item", "rating"])
    test_df = pd.read_csv(test_path, sep=' ', names=["user", "item", "rating"])
    num_users = max(train_df['user'].max(), test_df['user'].max()) + 1
    num_items = max(train_df['item'].max(), test_df['item'].max()) + 1

    user_tensor = torch.tensor(train_df['user'].values, dtype=torch.long)
    item_tensor = torch.tensor(train_df['item'].values, dtype=torch.long)
    edge_index = torch.stack([torch.cat([user_tensor, item_tensor + num_users]),
                              torch.cat([item_tensor + num_users, user_tensor])])
    return edge_index, train_df, test_df, num_users, num_items

def get_user_positive_items(df):
    pos_items = defaultdict(set)
    for user, item in zip(df['user'], df['item']):
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

# -------------------------- Training Function -------------------------- #
def train_model(config):
    edge_index, train_df, test_df, num_users, num_items = load_data("./data/train.txt", "./data/test.txt")
    model = GATRec(num_users, num_items,
                   in_channels=config['in_channels'],
                   hidden_channels=config['hidden_channels'],
                   out_channels=config['out_channels'],
                   num_heads=config['num_heads'],
                   dropout=config['dropout'],
                   edge_dropout=config['edge_dropout'],
                   neg_slope=config['neg_slope'])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    for epoch in range(30):
        optimizer.zero_grad()
        user_emb, item_emb = model(edge_index)

        pos_u = torch.tensor(train_df['user'].values)
        pos_i = torch.tensor(train_df['item'].values)
        neg_i = torch.randint(0, num_items, (len(pos_u),))

        user_vecs = user_emb[pos_u]
        pos_item_vecs = item_emb[pos_i]
        neg_item_vecs = item_emb[neg_i]

        pos_scores = (user_vecs * pos_item_vecs).sum(dim=1)
        neg_scores = (user_vecs * neg_item_vecs).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        train_pos = get_user_positive_items(train_df)
        return evaluate(user_emb, item_emb, test_df, train_pos, k_list=[10])

# ----------------------- Hyperparameter Tuning ------------------------ #
def tune_hyperparameters():
    default_config = {
        "in_channels": 64,
        "hidden_channels": 64,
        "out_channels": 64,
        "num_heads": 2,
        "dropout": 0.2,
        "edge_dropout": 0.2,
        "neg_slope": 0.2,
        "lr": 0.005,
        "batch_size": 128,  # not used directly in full-batch training
        "weight_decay": 0.0
    }

    param_grid = {
        "in_channels": [32, 64, 128],
        "hidden_channels": [32, 64, 128],
        "out_channels": [32, 64, 128],
        "num_heads": [1, 2, 4],
        "dropout": [0.0, 0.2, 0.5],
        "edge_dropout": [0.0, 0.2, 0.5],
        "neg_slope": [0.1, 0.2, 0.3],
        "lr": [0.001, 0.005, 0.01],
        "batch_size": [64, 128, 256],
        "weight_decay": [0.0, 1e-4, 1e-3],
    }

    for param, values in param_grid.items():
        print(f"\n Tuning {param} ")
        for val in values:
            config = default_config.copy()
            config[param] = val
            print(f"→ {param} = {val}")
            metrics = train_model(config)
            print(f"@10 Metrics: {metrics[10]}")
            save_result(param, val, metrics[10])

def save_result(param_name, param_value, metrics, folder='./result/'):
    os.makedirs(folder, exist_ok=True)
    result_file = os.path.join(folder, f"tune_{param_name}.csv")
    df_row = pd.DataFrame([{**metrics, param_name: param_value}])
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(result_file, index=False)

if __name__ == "__main__":
    tune_hyperparameters()