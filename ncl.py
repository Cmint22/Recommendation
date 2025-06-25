import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

# --------- InfoNCE --------- #
def info_nce_loss(z1, z2, temp=0.2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    pos_score = torch.sum(z1 * z2, dim=-1) / temp
    ttl_score = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(ttl_score, labels)

# --------- Neighbor Dropout Augmentation --------- #
def neighbor_dropout(edge_index, dropout_ratio=0.2):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) >= dropout_ratio
    return edge_index[:, mask]

# --------- NCL Model --------- #
class NCL(nn.Module):
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
        all_x = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_x.append(x)
        x = torch.stack(all_x, dim=1).mean(dim=1)
        return x[:self.num_users], x[self.num_users:]

# --------- Data Loader --------- #
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep=' ', names=['user', 'item', 'rating'])
    test_df = pd.read_csv(test_path, sep=' ', names=['user', 'item', 'rating'])
    num_users = max(train_df['user'].max(), test_df['user'].max()) + 1
    num_items = max(train_df['item'].max(), test_df['item'].max()) + 1
    user_tensor = torch.tensor(train_df['user'].values)
    item_tensor = torch.tensor(train_df['item'].values)
    edge_index = torch.stack([torch.cat([user_tensor, item_tensor + num_users]),
                              torch.cat([item_tensor + num_users, user_tensor])])
    return edge_index, train_df, test_df, num_users, num_items

def get_user_pos(train_df):
    pos_dict = defaultdict(set)
    for u, i in zip(train_df['user'], train_df['item']):
        pos_dict[u].add(i)
    return pos_dict

# --------- Evaluation --------- #
def evaluate(user_emb, item_emb, test_df, train_pos, k_list=[10]):
    metrics = {k: {"HR": 0, "P": 0, "R": 0, "NDCG": 0} for k in k_list}
    scores = torch.matmul(user_emb, item_emb.T).cpu().numpy()

    for user in test_df['user'].unique():
        test_items = set(test_df[test_df['user'] == user]['item'])
        known_items = train_pos[user]
        scores_user = scores[user]
        scores_user[list(known_items)] = -np.inf
        rank = np.argsort(-scores_user)

        for k in k_list:
            top_k = rank[:k]
            hits = len(set(top_k) & test_items)
            metrics[k]["HR"] += int(hits > 0)
            metrics[k]["P"] += hits / k
            metrics[k]["R"] += hits / len(test_items)
            metrics[k]["NDCG"] += sum([1 / np.log2(i + 2) if rank[i] in test_items else 0 for i in range(k)])

    num_users = len(test_df['user'].unique())
    for k in metrics:
        for m in metrics[k]:
            metrics[k][m] /= num_users
    return metrics

# --------- Training --------- #
def train_ncl(config):
    edge_index, train_df, test_df, num_users, num_items = load_data("./data/train.txt", "./data/test.txt")
    model = NCL(num_users, num_items, embedding_size=config["embedding_size"], n_layers=config["n_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    train_pos = get_user_pos(train_df)

    for epoch in tqdm(range(100)):
        model.train()
        optimizer.zero_grad()

        # Two views (dropout)
        edge1 = neighbor_dropout(edge_index, dropout_ratio=0.2)
        edge2 = neighbor_dropout(edge_index, dropout_ratio=0.2)
        u1, i1 = model(edge1)
        u2, i2 = model(edge2)

        # Contrastive loss
        loss_ssl = info_nce_loss(u1, u2, temp=config["ssl_temp"]) + info_nce_loss(i1, i2, temp=config["ssl_temp"])

        # BPR
        users = torch.tensor(train_df['user'].values)
        pos_items = torch.tensor(train_df['item'].values)
        neg_items = torch.randint(0, num_items, (len(users),))
        u_e = u1[users]
        p_e = i1[pos_items]
        n_e = i1[neg_items]
        pos_scores = torch.sum(u_e * p_e, dim=1)
        neg_scores = torch.sum(u_e * n_e, dim=1)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        reg = (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / len(users)

        total_loss = loss_bpr + config["reg_weight"] * reg + config["ssl_reg"] * loss_ssl
        total_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        metrics = evaluate(user_emb.cpu(), item_emb.cpu(), test_df, train_pos)
        return metrics

    
# ------------------- Hyperparameter Tuning ------------------- #
def tune_ncl_hyperparams():
    default_config = {
        "embedding_size": 64,
        "n_layers": 3,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "reg_weight": 1e-4,
        "ssl_temp": 0.1,
        "ssl_reg": 1e-7,
        "hyper_layers": 1,
        "alpha": 1.5,
        "proto_reg": 1e-7,
        "num_clusters": 2000,
        "m_step": 1,
        "warmup_steps": 20,
        "tau": 0.005
    }

    param_grid = {
        "embedding_size": [32, 64, 128],
        "n_layers": [1, 2, 3],
        "lr": [0.001, 0.01, 0.1],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "reg_weight": [1e-5, 1e-4, 1e-3],
        "ssl_temp": [0.1, 0.2, 0.5],
        "ssl_reg": [1e-7, 1e-6, 1e-5],
        "hyper_layers": [1, 2, 3],
        "alpha": [1.0, 1.5, 2.0],
        "proto_reg": [1e-8, 1e-7, 1e-6],
        "num_clusters": [1000, 2000, 3000],
        "m_step": [1, 2, 3],
        "warmup_steps": [10, 20, 30],
        "tau": [0.001, 0.005, 0.01, 0.05, 0.1]
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
    # df.to_csv(result_file, index=False)

if __name__ == "__main__":
    tune_ncl_hyperparams()