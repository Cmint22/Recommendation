import torch
import torch.nn.functional as F
from torch_geometric.nn import LGConv
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import os

# ------------------------- LightGCN Model ------------------------- #
class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.convs = torch.nn.ModuleList([LGConv() for _ in range(num_layers)])
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        out = x
        for conv in self.convs:
            out = conv(out, edge_index)
            x += out
        return x[:self.user_embedding.num_embeddings], x[self.user_embedding.num_embeddings:]

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
    return edge_index, train_df, test_df, num_users, num_items

def get_user_positive_items(train_df):
    pos_items = defaultdict(set)
    for user, item in zip(train_df['user'], train_df['item']):
        pos_items[user].add(item)
    return pos_items

def evaluate(user_emb, item_emb, test_df, train_pos, k_list=[10]):
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
    edge_index, train_df, test_df, num_users, num_items = load_data("./data/train.txt", "./data/test.txt")
    model = LightGCN(num_users, num_items, embedding_dim=config['embedding_dim'], num_layers=config['num_layers'])
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        user_emb, item_emb = model(edge_index)
        pos_u = torch.tensor(train_df['user'].values, dtype=torch.long)
        pos_i = torch.tensor(train_df['item'].values, dtype=torch.long)
        num_samples = len(pos_u)

        # Handle multiple negative samples
        if config['n_neg'] == 1:
            neg_i = torch.randint(0, num_items, (num_samples,))
        else:
            neg_i = torch.randint(0, num_items, (num_samples, config['n_neg']))
        user_vecs = user_emb[pos_u]                         
        pos_item_vecs = item_emb[pos_i] 
                           
        if config['n_neg'] == 1:
            neg_item_vecs = item_emb[neg_i]                  
            neg_scores = (user_vecs * neg_item_vecs).sum(dim=-1)  
        else:
            neg_item_vecs = item_emb[neg_i]                  
            user_vecs_expand = user_vecs.unsqueeze(1)        
            neg_scores = (user_vecs_expand * neg_item_vecs).sum(dim=-1).mean(dim=1)  
        pos_scores = (user_vecs * pos_item_vecs).sum(dim=-1)     

        if config['loss_type'] == "bpr":
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        elif config['loss_type'] == "bce":
            scores = torch.matmul(user_vecs, item_emb.t())
            labels = torch.zeros_like(scores)
            labels[range(len(pos_u)), pos_i] = 1.0
            loss = F.binary_cross_entropy_with_logits(scores, labels)
        else:
            raise ValueError("Unsupported loss_type")

        # Regularization
        loss += config['reg_weight'] * (user_vecs.norm(2).pow(2) + pos_item_vecs.norm(2).pow(2))
        loss.backward()
        optimizer.step()


    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        train_pos = get_user_positive_items(train_df)
        metrics = evaluate(user_emb, item_emb, test_df, train_pos, k_list=[10])
        return metrics

# ------------------------- Hyperparameter Tuning ------------------------- #
def tune_hyperparameters():
    default_config = {
        "embedding_dim": 64,
        "num_layers": 3,
        "reg_weight": 1e-4,
        "weight_decay": 0.0,
        "n_neg": 1,
        "loss_type": "bpr",
        "optimizer": "Adam",
        "lr": 0.01
    }

    param_grid = {
        "embedding_dim": [32, 64, 128],
        "num_layers": [1, 2, 3, 4],
        "reg_weight": [1e-4, 1e-3, 1e-2],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "n_neg": [1, 3, 5],
        "loss_type": ["bpr", "bce"],
        "optimizer": ["Adam", "AdamW", "SGD"],
        "lr": [0.001, 0.01, 0.05]
    }

    for param_name, values in param_grid.items():
        print(f"Tuning {param_name}")
        for val in values:
            config = default_config.copy()
            config[param_name] = val
            print(f"→ {param_name} = {val}")
            metrics = train_model(config)
            print(f"Result @10: {metrics[10]}\n")

if __name__ == "__main__":
    tune_hyperparameters()
