import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os

# --------- DropEdge --------- #
def edge_dropout(edge_index, drop_rate=0.2):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) >= drop_rate
    return edge_index[:, mask]

# --------- InfoNCE Loss --------- #
def info_nce_loss(z1, z2, temp=0.2):
    z1, z2 = F.normalize(z1), F.normalize(z2)
    sim = torch.mm(z1, z2.t()) / temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim, labels)

# --------- GCL Model --------- #
class GCLModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64, num_layers=1):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.convs = nn.ModuleList([SAGEConv(emb_size, emb_size) for _ in range(num_layers)])
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index)
        return x[:self.user_emb.num_embeddings], x[self.user_emb.num_embeddings:]

# --------- Load Data --------- #
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

# --------- Evaluation --------- #
def get_user_pos(train_df):
    pos = defaultdict(set)
    for u, i in zip(train_df['user'], train_df['item']):
        pos[u].add(i)
    return pos

def evaluate(user_emb, item_emb, test_df, train_pos, k=10):
    scores = torch.matmul(user_emb, item_emb.T).cpu().numpy()
    metrics = {"HR": 0, "P": 0, "R": 0, "NDCG": 0}
    users = test_df['user'].unique()
    for user in users:
        test_items = set(test_df[test_df['user'] == user]['item'])
        known_items = train_pos[user]
        scores_user = scores[user]
        scores_user[list(known_items)] = -np.inf
        rank = np.argsort(-scores_user)[:k]
        hits = len(set(rank) & test_items)
        metrics["HR"] += int(hits > 0)
        metrics["P"] += hits / k
        metrics["R"] += hits / len(test_items)
        metrics["NDCG"] += sum([1 / np.log2(i + 2) if rank[i] in test_items else 0 for i in range(k)])
    for key in metrics:
        metrics[key] /= len(users)
    return metrics

# --------- Training --------- #
def train_gcl(config):
    edge_index, train_df, test_df, num_users, num_items = load_data('./data/train.txt', './data/test.txt')
    model = GCLModel(num_users, num_items, emb_size=config["embedding_size"], num_layers=config["num_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    train_pos = get_user_pos(train_df)

    for epoch in tqdm(range(100)):
        model.train()
        optimizer.zero_grad()
        edge1 = edge_dropout(edge_index, 0.2)
        edge2 = edge_dropout(edge_index, 0.2)
        z1_u, z1_i = model(edge1)
        z2_u, z2_i = model(edge2)
        ssl_loss = info_nce_loss(z1_u, z2_u, config["ssl_temp"]) + info_nce_loss(z1_i, z2_i, config["ssl_temp"])

        users = torch.tensor(train_df['user'].values)
        pos_items = torch.tensor(train_df['item'].values)
        neg_items = torch.randint(0, num_items, (len(users),))
        u_e = z1_u[users]
        p_e = z1_i[pos_items]
        n_e = z1_i[neg_items]
        pos_scores = (u_e * p_e).sum(1)
        neg_scores = (u_e * n_e).sum(1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / len(users)
        total_loss = bpr_loss + config["reg_weight"] * reg + config["ssl_reg"] * ssl_loss
        total_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        return evaluate(user_emb.cpu(), item_emb.cpu(), test_df, train_pos)

# --------- Hyperparam Search --------- #
def tune_gcl_hyperparams():
    default_config = {
        "embedding_size": 64,
        "num_layers": 1,
        "reg_weight": 1e-4,
        "lr": 0.001,
        "weight_decay": 0.0,
        "ssl_temp": 0.2,
        "ssl_reg": 1e-6
    }

    param_grid = {
        "embedding_size": [32, 64, 128],
        "num_layers": [0, 1, 2],
        "reg_weight": [1e-5, 1e-4, 1e-3],
        "lr": [0.0001, 0.001, 0.01],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "ssl_temp": [0.1, 0.2, 0.3],
        "ssl_reg": [1e-7, 1e-6, 1e-5]
    }

    for param_name, values in param_grid.items():
        print(f"Tuning {param_name}")
        for val in values:
            config = default_config.copy()
            config[param_name] = val
            print(f" → {param_name} = {val}")
            metrics = train_gcl(config)
            print(f"Result @10: {metrics}\n")
            save_result(param_name, val, metrics)
    
def save_result(param_name, param_value, metrics, folder='./result/'):
    os.makedirs(folder, exist_ok=True)
    result_file = os.path.join(folder, f"gcl_tune_{param_name}.csv")
    df_row = pd.DataFrame([{**metrics, param_name: param_value}])
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(result_file, index=False)

if __name__ == "__main__":
    tune_gcl_hyperparams()