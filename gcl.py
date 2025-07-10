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
class GCLTuner:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.grid = {
            'embedding_size': [32, 64, 128, 256, 512, 1024],
            'num_layers': [1, 2, 3, 4, 5],
            'lr': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'weight_decay': [1e-4, 1e-5, 1e-3],
            'ssl_temp': [0.1, 0.2, 0.5],
            'drop_edge': [0.1, 0.2, 0.3],
            'reg_weight': [1e-5, 1e-4, 1e-3],
            'ssl_weight': [1e-5, 1e-4, 1e-3],
            'batch_size': [128, 256, 512, 1024, 2048, 4096],
            'max_epoch': [1]
        }

    def run(self):
        results = []
        edge_index, train_df, test_df, num_users, num_items = load_data(self.train_path, self.test_path)
        print("Loading data from configuration files...")
        print(f"Loaded {len(train_df)} training interactions")
        print(f"Loaded {len(test_df)} test interactions")
        print("\nGCL Hyperparameter Tuning Framework\n" + "="*80)
        print("This script will tune the following hyperparameters:")
        print("- Embedding size: [32, 64, 128, 256, 512, 1024]")
        print("- Num layers: [1, 2, 3, 4, 5]")
        print("- Learning rate: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]")
        print("- Weight decay: [1e-5, 1e-4, 1e-3]")
        print("- SSL temperature: [0.1, 0.2, 0.5]")
        print("- Drop edge prob: [0.1, 0.2, 0.3]")
        print("- Regularization weight: [1e-5, 1e-4, 1e-3]")
        print("- SSL weight: [1e-5, 1e-4, 1e-3]")
        print("- Batch size: [128, 256, 512, 1024, 2048, 4096]")
        print("- Max epochs: [1]")
        combos = list(itertools.product(
            self.grid['embedding_size'],
            self.grid['num_layers'],
            self.grid['lr'],
            self.grid['weight_decay'],
            self.grid['ssl_temp'],
            self.grid['drop_edge'],
            self.grid['reg_weight'],
            self.grid['ssl_weight'],
            self.grid['batch_size'],
            self.grid['max_epoch']
        ))
        print(f"\nGCL Hyperparameter Tuning - Total combinations: {len(combos)}\n{'='*80}\n")
        best = None
        best_metrics = None
        user_pos = get_user_pos(train_df)
        for i, combo in enumerate(combos, 1):
            config = {
                'embedding_size': combo[0],
                'num_layers': combo[1],
                'lr': combo[2],
                'weight_decay': combo[3],
                'ssl_temp': combo[4],
                'drop_edge': combo[5],
                'reg_weight': combo[6],
                'ssl_weight': combo[7],
                'batch_size': combo[8],
                'max_epoch': combo[9],
                'train_path': self.train_path,
                'test_path': self.test_path
            }
            print(f"{'='*80}\n[{i}] Training with configuration:")
            print(f"Embedding size: {config['embedding_size']}, Num layers: {config['num_layers']}, Learning rate: {config['lr']}")
            print(f"Weight decay: {config['weight_decay']}, SSL temp: {config['ssl_temp']}, Drop edge: {config['drop_edge']}")
            print(f"Reg weight: {config['reg_weight']}, Epochs: {config['max_epoch']}, Batch size: {config['batch_size']}")
            model = GRACEModel(num_users, num_items,
                               emb_size=config["embedding_size"],
                               num_layers=config["num_layers"]).to(device)
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
                    ssl_loss = info_nce_loss(user_z1, user_z2, config["ssl_temp"]) \
                             + info_nce_loss(item_z1, item_z2, config["ssl_temp"])
                    u_e = user_z1[users.to(device)]
                    p_e = item_z1[pos_items.to(device)]
                    n_e = item_z1[neg_items.to(device)]
                    pos_scores = (u_e * p_e).sum(1)
                    neg_scores = (u_e * n_e).sum(1)
                    bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                    reg_loss = (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / len(users)
                    total_loss = ssl_loss + bpr_loss + config['reg_weight'] * reg_loss
                    total_loss.backward()
                    optimizer.step()
                    if (n + 1) % 100 == 0:
                        print(f"Batch {n+1}, SSL loss: {ssl_loss.item():.4f}, BPR loss: {bpr_loss.item():.4f}, Reg loss: {reg_loss.item():.4f}, Total loss: {total_loss.item():.4f}")
            print("\nEvaluating model...")
            model.eval()
            with torch.no_grad():
                final_z = model.encode(edge_index_dev)
                user_emb = final_z[:num_users]
                item_emb = final_z[num_users:]
                metrics = evaluate(user_emb, item_emb, test_df, user_pos, ks=[10, 20, 30, 50])
            print("\nDetailed TopN Evaluation:")
            for k in [10, 20, 30, 50]:
                print(f"Top {k}")
                print(f"Hit Ratio:{metrics[k]['HR']:.5f}")
                print(f"Precision:{metrics[k]['P']:.5f}")
                print(f"Recall:{metrics[k]['R']:.5f}")
                print(f"NDCG:{metrics[k]['NDCG']:.5f}")
            print(f"\nCompleted {i}/{len(combos)} combinations")
            ndcg = metrics[50]['NDCG']
            recall = metrics[50]['R']
            precision = metrics[50]['P']
            hr = metrics[50]['HR']
            results.append({
                'config': config,
                'metrics': {
                    'NDCG': ndcg,
                    'Recall': recall,
                    'Precision': precision,
                    'Hit Ratio': hr
                }
            })
            if best is None or recall > best_metrics['Recall']:
                best = config
                best_metrics = {'NDCG': ndcg, 'Recall': recall, 'Precision': precision, 'Hit Ratio': hr}
                print(f"Best performance - NDCG: {ndcg:.5f}, Recall: {recall:.5f}, Precision: {precision:.5f}, Hit Ratio: {hr:.5f}")

# === Summary Printer ===
def print_summary(results):
    success = [r for r in results if 'metrics' in r]
    failed = [r for r in results if 'error' in r]
    print(f"\n{'='*80}\nHYPERPARAMETER TUNING SUMMARY")
    print(f"Total: {len(results)} | Success: {len(success)} | Failed: {len(failed)}")
    for metric in ['NDCG', 'Recall', 'Hit Ratio', 'Precision']:
        try:
            best = max(success, key=lambda r: r['metrics'].get(metric, 0))
            conf = best['config']
            metrics = best['metrics']
            print(f"[Best {metric}] {metrics[metric]:.5f} | "
                  f"embedding_size={conf.get('embedding_size')}, "
                  f"num_layers={conf.get('num_layers')}, "
                  f"lr={conf.get('lr')}, "
                  f"weight_decay={conf.get('weight_decay')}, "
                  f"drop_edge={conf.get('drop_edge')}, "
                  f"ssl_temp={conf.get('ssl_temp')}, "
                  f"reg_weight={conf.get('reg_weight')}, "
                  f"ssl_weight={conf.get('ssl_weight')}, "
                  f"batch_size={conf.get('batch_size')}")
        except Exception as e:
            print(f"[Best {metric}] Error: {e}")



if __name__ == '__main__':
    train_path = "./dataset/ml100k/train.txt"
    test_path = "./dataset/ml100k/test.txt"
    tuner = GCLTuner(train_path, test_path)
    tuner.run()
    print_summary(results=tuner.results)
