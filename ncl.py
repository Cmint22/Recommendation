import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from collections import defaultdict
import numpy as np
import pandas as pd
import os

# ------------------- Graph Encoder (LGConv-based) ------------------- #
class NCLModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        out = torch.mean(all_embeddings, dim=1)
        return out[:self.num_users], out[self.num_items:], all_embeddings
    
# ------------------- Load Data ------------------- #
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
            all_metrics[k]["HR"] += hits / len(test_items)
            all_metrics[k]["P"] += hits / k
            all_metrics[k]["R"] += hits / len(test_items)
            all_metrics[k]["NDCG"] += ndcg

        num_users = len(test_df['user'].unique())
        for k in all_metrics:
            for m in all_metrics[k]:
                all_metrics[k][m] /= num_users
        return all_metrics

   
def train_ncl(config):
    edge_index, train_df, test_df, num_users, num_items = load_data("./data/train.txt", "./data/test.txt")
    model = NCLModel(num_users, num_items, embedding_size=config['embedding_size'], n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        user_emb, item_emb, emb_list = model(edge_index)

        users = torch.tensor(train_df['user'].values, dtype=torch.long)
        pos_items = torch.tensor(train_df['item'].values, dtype=torch.long)
        num_samples = len(users)

        neg_items = torch.randit(0, num_items, (num_samples,))
        u_e = user_emb[users]
        p_e = item_emb[pos_items]
        n_e = item_emb[neg_items]

        pos_scores = torch.sum(u_e * p_e, dim=1)
        neg_scores = torch.sum(u_e * n_e, dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        reg = u_e.norm(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)
        bpr_loss += config['reg_weight'] * reg

        bpr_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        user_emb, item_emb, _ = model(edge_index)
        user_emb, item_emb = user_emb.cpu(), item_emb.cpu()
        train_pos = get_user_positive_items(train_df)
        metrics = evaluate(user_emb, item_emb, test_df, train_df)
        return metrics
    
def tune_ncl_hyperparams():
    default_config = {
        "embedding_size": 64,
        "n_layers": 3,
        "reg_weight": 1e-4,
        "ssl_temp": 0.1,
        "ssl_reg": 1e-7,
        "hyper_layers": 1,
        "alpha": 1,
        "proto_reg": 8e-8,
        "num_clusters": 1000,
        "m_step": 1,
        "warmup_steps": 20
    }

    param_grid = {
        "embedding_size": [32, 64, 128],
        "n_layers": [1, 2, 3],
        "reg_weight": [1e-5, 1e-4, 1e-3],
        "ssl_temp": [0.2, 0.2, 0.3],
        "ssl_reg": [1e-7, 1e-5, 1e-5],
        "hyper_layers": [1, 2, 3],
        "alpha": [ 0.5, 1, 1.5],
        "proto_reg": [8e-8, 1e-7, 1e-6],
        "num_clusters": [500, 1000, 2000],
        "m_step": [1, 2, 3],
        "warmup_steps": [10, 20, 30]
    }

    for param_name, values in param_grid.items():
        print(f"Tuning {param_name}")
        for val in values:
            config = default_config.copy()
            config[param_name] = val
            print(f" → {param_name} = {val}")
            metrics = train_ncl(config)
            print(f"Result @10: {metrics[10]}\n")
            save_result(param_name, val, metrics[10])

def save_result(param_name, param_value, metrics, folder='./result/'):
    os.makedirs(folder, exist_ok=True)
    result_file = os.path.join(folder, f"ncl_tune_{param_name}.csv")
    df_row = pd.DataFrame([{**metrics, param_name: param_value}])
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(result_file, index=False)


if __name__ == "__main__":
    tune_ncl_hyperparams()                      
