import os
import json
import copy
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
from random import shuffle
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== EdgeRemoving Augmentation =====
class EdgeRemoving:
    def __init__(self, pe=0.2):
        self.pe = pe

    def __call__(self, edge_index):
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) >= self.pe
        return edge_index[:, keep_mask]

# ===== InfoNCE Loss =====
def info_nce_loss(z1, z2, temp=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.t()) / temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss1 = F.cross_entropy(sim, labels)
    loss2 = F.cross_entropy(sim.T, labels)
    return (loss1 + loss2) / 2

# ===== GRACE Model =====
class GRACEModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64, num_layers=2, proj_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.convs = nn.ModuleList([nn.Linear(emb_size, emb_size) for _ in range(num_layers)])
        self.proj_head = nn.Sequential(
            nn.Linear(emb_size, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def encode(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        for conv in self.convs:
            x = conv(x)
        return x

    def project(self, x):
        return self.proj_head(x)

    def forward(self, edge_index1, edge_index2):
        z1 = self.project(self.encode(edge_index1))
        z2 = self.project(self.encode(edge_index2))
        return z1, z2

# ===== Data Loader =====
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep=' ', names=['user', 'item', 'rating'])
    test_df = pd.read_csv(test_path, sep=' ', names=['user', 'item', 'rating'])
    num_users = max(train_df['user'].max(), test_df['user'].max()) + 1
    num_items = max(train_df['item'].max(), test_df['item'].max()) + 1
    user_tensor = torch.tensor(train_df['user'].values)
    item_tensor = torch.tensor(train_df['item'].values)
    edge_index = torch.stack([
        torch.cat([user_tensor, item_tensor + num_users]),
        torch.cat([item_tensor + num_users, user_tensor])
    ])
    return edge_index, train_df, test_df, num_users, num_items

# ===== Evaluation =====
def get_user_pos(train_df):
    pos = defaultdict(set)
    for u, i in zip(train_df['user'], train_df['item']):
        pos[u].add(i)
    return pos

def evaluate(user_emb, item_emb, test_df, train_pos, ks=[10, 20, 30, 50]):
    scores = torch.matmul(user_emb, item_emb.T).cpu().numpy()
    metrics = {k: {"HR": 0, "P": 0, "R": 0, "NDCG": 0} for k in ks}
    users = test_df['user'].unique()
    for user in users:
        test_items = set(test_df[test_df['user'] == user]['item'])
        known_items = train_pos[user]
        scores_user = scores[user]
        scores_user[list(known_items)] = -np.inf
        rank = np.argsort(-scores_user)
        for k in ks:
            topk = rank[:k]
            hits = len(set(topk) & test_items)
            ndcg = sum([1 / np.log2(i + 2) if topk[i] in test_items else 0 for i in range(k)])
            metrics[k]["HR"] += int(hits > 0)
            metrics[k]["P"] += hits / k
            metrics[k]["R"] += hits / len(test_items)
            metrics[k]["NDCG"] += ndcg
    for k in ks:
        for key in metrics[k]:
            metrics[k][key] /= len(users)
    return metrics

# ===== Batch Sampler for BPR =====
def next_batch_pairwise(train_df, batch_size, num_users, num_items, user_pos):
    idx = np.arange(len(train_df))
    np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start+batch_size]
        users = train_df.iloc[batch_idx]['user'].values
        pos_items = train_df.iloc[batch_idx]['item'].values
        neg_items = []
        for u in users:
            while True:
                ni = np.random.randint(0, num_items)
                if ni not in user_pos[u]:
                    neg_items.append(ni)
                    break
        yield torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)


# ===== GCL Trainer with Grid Search =====
def generate_independent_grid(defaults, tuning_grid):
    grid = []
    for key, values in tuning_grid.items():
        for val in values:
            config = copy.deepcopy(defaults)
            config[key] = val
            grid.append(config)
    return grid
class GCLTuner:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.default_config = {
            'embedding_size': 128,
            'num_layers': 2,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'ssl_temp': 0.5,
            'drop_edge': 0.2,
            'reg_weight': 1e-4,
            'ssl_weight': 0.1,
            'batch_size': 2048,
            'max_epoch': 1
        }
        self.tuning_grid = {
            'embedding_size': [128, 256, 512, 1024],
            'num_layers': [1, 2, 3, 4],
            'lr': [0.0001, 0.001, 0.01, 0.1],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'ssl_temp': [0.2, 0.5, 0.8],
            'drop_edge': [0.1, 0.2, 0.3],
            'reg_weight': [1e-5, 1e-4, 1e-3],
            'ssl_weight': [0.05, 0.1, 0.2],
            'batch_size': [256, 512, 1024, 2048]
        }
        self.results = []

    def run(self):
        edge_index, train_df, test_df, num_users, num_items = load_data(self.train_path, self.test_path)
        user_pos = get_user_pos(train_df)
        all_configs = generate_independent_grid(self.default_config, self.tuning_grid)
        total_configs = len(all_configs)
        print(f"\nTotal hyperparameter combinations to train: {total_configs}")
        for i, config in enumerate(all_configs, 1):
            print(f"\n{'='*80}")
            # print(f"[{i}/{total_configs}] Training with configuration:")
            changed_param = [(k, v) for k, v in config.items() if self.default_config.get(k) != v]
            changed_str = ', '.join(f"{k}={v}" for k, v in changed_param)
            print(f"[{i}/{total_configs}] Training with configuration: {changed_str}")
            for k, v in config.items():
                print(f"{k}: {v}")
            model = GRACEModel(num_users, num_items, emb_size=config["embedding_size"], num_layers=config["num_layers"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            aug = EdgeRemoving(pe=config['drop_edge'])
            edge_index_dev = edge_index.to(device)
            model.train()
            for epoch in range(config['max_epoch']):
                batch_iter = next_batch_pairwise(train_df, config['batch_size'], num_users, num_items, user_pos)
                for n, (users, pos_items, neg_items) in enumerate(batch_iter):
                    optimizer.zero_grad()
                    edge1 = aug(edge_index_dev)
                    edge2 = aug(edge_index_dev)
                    z1, z2 = model(edge1, edge2)
                    user_z1, item_z1 = z1[:num_users], z1[num_users:]
                    user_z2, item_z2 = z2[:num_users], z2[num_users:]
                    ssl_loss = info_nce_loss(user_z1, user_z2, config["ssl_temp"]) + info_nce_loss(item_z1, item_z2, config["ssl_temp"])
                    u_e = user_z1[users.to(device)]
                    p_e = item_z1[pos_items.to(device)]
                    n_e = item_z1[neg_items.to(device)]
                    pos_scores = (u_e * p_e).sum(1)
                    neg_scores = (u_e * n_e).sum(1)
                    bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                    reg_loss = (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / len(users)
                    total_loss = config['ssl_weight'] * ssl_loss + bpr_loss + config['reg_weight'] * reg_loss
                    total_loss.backward()
                    optimizer.step()
                    if (n + 1) % 100 == 0:
                        print(f"Batch {n+1} | SSL: {ssl_loss.item():.4f}, BPR: {bpr_loss.item():.4f}, Reg: {reg_loss.item():.4f}, Total: {total_loss.item():.4f}")
            print("\nEvaluating...")
            model.eval()
            with torch.no_grad():
                final_z = model.encode(edge_index_dev)
                user_emb = final_z[:num_users]
                item_emb = final_z[num_users:]
                metrics = evaluate(user_emb, item_emb, test_df, user_pos, ks=[10, 20, 30, 50])
            print("\nTop-N Metrics:")
            for k in [10, 20, 30, 50]:
                print(f"Top {k} | HR: {metrics[k]['HR']:.4f} | P: {metrics[k]['P']:.4f} | R: {metrics[k]['R']:.4f} | NDCG: {metrics[k]['NDCG']:.4f}")
            self.results.append({'config': config, 'metrics': metrics})

# # === Summary Printer ===
# def print_summary(results):
#     success = [r for r in results if 'metrics' in r]
#     failed = [r for r in results if 'error' in r]
#     print(f"\n{'='*80}\nHYPERPARAMETER TUNING SUMMARY")
#     print(f"Total: {len(results)} | Success: {len(success)} | Failed: {len(failed)}")
#     for metric in ['NDCG', 'Recall', 'Hit Ratio', 'Precision']:
#         try:
#             best = max(success, key=lambda r: r['metrics'].get(metric, 0))
#             conf = best['config']
#             metrics = best['metrics']
#             print(f"[Best {metric}] {metrics[metric]:.5f} | "
#                   f"embedding_size={conf.get('embedding_size')}, "
#                   f"num_layers={conf.get('num_layers')}, "
#                   f"lr={conf.get('lr')}, "
#                   f"weight_decay={conf.get('weight_decay')}, "
#                   f"drop_edge={conf.get('drop_edge')}, "
#                   f"ssl_temp={conf.get('ssl_temp')}, "
#                   f"reg_weight={conf.get('reg_weight')}, "
#                   f"ssl_weight={conf.get('ssl_weight')}, "
#                   f"batch_size={conf.get('batch_size')}")
#         except Exception as e:
#             print(f"[Best {metric}] Error: {e}")


if __name__ == '__main__':
    train_path = "./dataset/ml100k/train.txt"
    test_path = "./dataset/ml100k/test.txt"
    print("Loading data from configuration files...")
    print(f"Loaded {len(train_path)} training interactions")
    print(f"Loaded {len(test_path)} test interactions")
    print("\nNCL Hyperparameter Tuning Framework\n" + "="*80)
    tuner = GCLTuner(train_path, test_path)
    tuner.run()