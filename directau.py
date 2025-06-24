import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
import os

# ------------------------- DirectAU Model ------------------------- #
class DirectAU(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64, gamma=1.0):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.gamma = gamma
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def alignment(self, u, i):
        u, i = F.normalize(u, dim=-1), F.normalize(i, dim=-1)
        return ((u - i).pow(2).sum(dim=1)).mean()

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8)

    def calculate_loss(self, user_vecs, item_vecs):
        align = self.alignment(user_vecs, item_vecs)
        uniform = self.gamma * (self.uniformity(user_vecs) + self.uniformity(item_vecs)) / 2
        return align + uniform

# ------------------------- Data Loader ------------------------- #
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep=' ', names=["user", "item", "rating"])
    test_df = pd.read_csv(test_path, sep=' ', names=["user", "item", "rating"])
    num_users = max(train_df['user'].max(), test_df['user'].max()) + 1
    num_items = max(train_df['item'].max(), test_df['item'].max()) + 1
    return train_df, test_df, num_users, num_items

def get_user_positive_items(df):
    pos_dict = defaultdict(set)
    for u, i in zip(df['user'], df['item']):
        pos_dict[u].add(i)
    return pos_dict

# ------------------------- Evaluation ------------------------- #
def evaluate(user_emb, item_emb, test_df, train_pos, k_list=[10]):
    metrics = {k: {"HR": 0, "P": 0, "R": 0, "NDCG": 0} for k in k_list}
    scores = torch.matmul(user_emb, item_emb.T)

    for user in test_df['user'].unique():
        test_items = set(test_df[test_df['user'] == user]['item'])
        if not test_items:
            continue
        known_items = train_pos[user]
        scores_u = scores[user].detach().numpy()
        scores_u[list(known_items)] = -np.inf
        rank = np.argsort(-scores_u)

        for k in k_list:
            top_k = rank[:k]
            hits = len(set(top_k) & test_items)
            metrics[k]["HR"] += int(hits > 0)
            metrics[k]["P"] += hits / k
            metrics[k]["R"] += hits / len(test_items)
            metrics[k]["NDCG"] += sum([
                1 / np.log2(i + 2) if rank[i] in test_items else 0
                for i in range(k)
            ])

    num_users = len(test_df['user'].unique())
    for k in metrics:
        for m in metrics[k]:
            metrics[k][m] /= num_users
    return metrics

# ------------------------- Training ------------------------- #
def train_model(config):
    train_df, test_df, num_users, num_items = load_data("./data/train.txt", "./data/test.txt")
    train_pos = get_user_positive_items(train_df)

    model = DirectAU(num_users, num_items, embedding_size=config['embedding_size'], gamma=config['gamma'])
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    user_tensor = torch.tensor(train_df['user'].values, dtype=torch.long)
    item_tensor = torch.tensor(train_df['item'].values, dtype=torch.long)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        user_emb, item_emb = model()
        user_vecs = user_emb[user_tensor]
        item_vecs = item_emb[item_tensor]
        loss = model.calculate_loss(user_vecs, item_vecs)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model()
        metrics = evaluate(user_emb, item_emb, test_df, train_pos)
        return metrics

# ------------------------- Tuning ------------------------- #
def tune_hyperparameters():
    default_config = {
        "embedding_size": 64,
        "gamma": 1.0,
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 1e-6
    }

    param_grid = {
        "embedding_size": [32, 64, 128],
        "gamma": [0.1, 1.0, 5.0],
        "lr": [0.0001, 0.001, 0.01],
        "weight_decay": [1e-6, 1e-5, 1e-4],
        "optimizer": ["Adam", "SGD"]
    }

    for param_name, values in param_grid.items():
        for val in values:
            config = default_config.copy()
            config[param_name] = val
            print(f"→ Tuning {param_name} = {val}")
            metrics = train_model(config)
            print(f"Result @10: {metrics[10]}\n")
            save_result(param_name, val, metrics[10])

def save_result(param_name, val, metrics, folder='./result/'):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"directau_tune_{param_name}.csv")
    row = pd.DataFrame([{**metrics, param_name: val}])
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    tune_hyperparameters()