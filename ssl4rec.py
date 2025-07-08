import os, json, torch, math, heapq, itertools, copy
from unittest import result
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from collections import defaultdict
from random import shuffle, choice
from time import time, localtime, strftime
from numpy.linalg import norm
from numba import jit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ====================== Losses =========================
def l2_reg_loss(reg, *args):
    return sum(torch.norm(x, p=2) / x.shape[0] for x in args) * reg

def InfoNCE(view1, view2, temperature, b_cos=True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 @ view2.T) / temperature
    return -torch.diag(F.log_softmax(pos_score, dim=1)).mean()

def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.exp(user_emb @ item_emb.T / temperature).sum(dim=1)
    return (-torch.log(pos_score / ttl_score + 1e-6)).mean()

# ====================== Sampler =========================
def next_batch_pairwise(data, batch_size, n_negs=1):
    shuffle(data.training_data)
    ptr, size = 0, len(data.training_data)
    items = list(data.item.keys())
    while ptr < size:
        end = min(ptr + batch_size, size)
        users, items_batch = zip(*[(data.training_data[i][0], data.training_data[i][1]) for i in range(ptr, end)])
        u_idx, i_idx, j_idx = [], [], []
        for u, i in zip(users, items_batch):
            uid, iid = data.user[u], data.item[i]
            u_idx.append(uid); i_idx.append(iid)
            for _ in range(n_negs):
                j = choice(items)
                while j in data.training_set_u[u]:
                    j = choice(items)
                j_idx.append(data.item[j])
        yield u_idx, i_idx, j_idx
        ptr = end

# ====================== Data Loader =========================
def load_data(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [[*line.strip().split()[:2], 1.0] for line in f if line.strip()]

# ====================== Interaction =========================
class Interaction:
    def __init__(self, training_data, test_data):
        self.training_data, self.test_data = training_data, test_data
        self.user, self.item = {}, {}
        self.id2user, self.id2item = {}, {}
        self.training_set_u, self.training_set_i = defaultdict(dict), defaultdict(dict)
        self.test_set, self.test_set_item = defaultdict(dict), set()
        self._build_mappings()
        self.user_num, self.item_num = len(self.user), len(self.item)
        self.norm_adj = self._create_normalized_adj()
    def _build_mappings(self):
        for u, i, r in self.training_data:
            if u not in self.user:
                uid = len(self.user); self.user[u] = uid; self.id2user[uid] = u
            if i not in self.item:
                iid = len(self.item); self.item[i] = iid; self.id2item[iid] = i
            self.training_set_u[u][i] = r; self.training_set_i[i][u] = r
        for u, i, r in self.test_data:
            if u in self.user and i in self.item:
                self.test_set[u][i] = r; self.test_set_item.add(i)
    def _create_normalized_adj(self):
        u_np = np.array([self.user[u] for u, _, _ in self.training_data])
        i_np = np.array([self.item[i] + self.user_num for _, i, _ in self.training_data])
        data = np.ones_like(u_np, dtype=np.float32)
        adj = sp.csr_matrix((data, (u_np, i_np)), shape=(self.user_num + self.item_num,)*2)
        adj = adj + adj.T
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5); d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat = sp.diags(d_inv_sqrt)
        return d_mat.dot(adj).dot(d_mat).tocsr()
    def get_user_id(self, u): return self.user.get(u)
    def get_item_id(self, i): return self.item.get(i)
    def user_rated(self, u): return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

# ====================== Evaluation =========================
@jit(nopython=True)
def find_k_largest(K, scores):
    heap = [(s, i) for i, s in enumerate(scores[:K])]
    heapq.heapify(heap)
    for i, s in enumerate(scores[K:], K):
        if s > heap[0][0]:
            heapq.heapreplace(heap, (s, i))
    heap.sort(key=lambda x: -x[0])
    return [i for _, i in heap], [s for s, _ in heap]

def ranking_evaluation(origin, res, N):
    def hits(): return {u: len(set(origin[u]) & {i for i, _ in res[u][:n]}) for u in res}
    def precision(h): return round(sum(h.values()) / (len(h) * n), 5)
    def recall(h): return round(sum(h[u] / len(origin[u]) for u in h) / len(h), 5)
    def ndcg():
        total = 0
        for u in res:
            DCG = sum(1 / math.log2(r + 2) if i in origin[u] else 0 for r, (i, _) in enumerate(res[u][:n]))
            IDCG = sum(1 / math.log2(r + 2) for r in range(min(n, len(origin[u]))))
            total += DCG / IDCG if IDCG > 0 else 0
        return round(total / len(res), 5)
    report = []
    for n in N:
        h = hits()
        report.append(f"Top {n}\n")
        report.append(f"Hit Ratio:{round(sum(h.values()) / sum(len(origin[u]) for u in origin), 5)}\n")
        report.append(f"Precision:{precision(h)}\n")
        report.append(f"Recall:{recall(h)}\n")
        report.append(f"NDCG:{ndcg()}\n")
    return report

# ====================== Recommender =========================
class Recommender:
    def __init__(self, conf, train, test):
        self.config = conf
        self.data = Interaction(train, test)
        self.emb_size = conf['embedding.size']
        self.batch_size = conf['batch.size']
        self.lRate = conf['learning.rate']
        self.reg = conf['reg.lambda']
        self.maxEpoch = conf.get('max.epoch', 1)
        self.topN = conf.get('item.ranking.topN', [10])
        self.max_N = max(self.topN)

# ====================== GraphRecommender =========================
class GraphRecommender(Recommender):
    def __init__(self, conf, train, test):
        super().__init__(conf, train, test)
        self.bestPerformance = []
    def test(self):
        rec_list = {}
        for u in self.data.test_set:
            uid = self.data.get_user_id(u)
            scores = self.predict(u)
            rated = self.data.user_rated(u)[0]
            for i in rated:
                scores[self.data.get_item_id(i)] = -1e8
            idxs, vals = find_k_largest(self.max_N, scores)
            rec_list[u] = [(self.data.id2item[i], v) for i, v in zip(idxs, vals)]
        return rec_list
    def evaluate(self, rec_list):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print(f"{'='*80}\nEvaluation:\n{''.join(metrics)}")
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.strip().split(':')]}



class DNNEncoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, tau, n_layers):
        super().__init__()
        self.emb_size = emb_size
        self.tau = tau
        self.dropout = nn.Dropout(drop_rate)
        self.n_layers = n_layers
        init = nn.init.xavier_uniform_
        self.initial_user = nn.Parameter(init(torch.empty(data.user_num, emb_size, device=device)))
        self.initial_item = nn.Parameter(init(torch.empty(data.item_num, emb_size, device=device)))
        self.user_net = self.build_mlp(emb_size)
        self.item_net = self.build_mlp(emb_size)
        self.to(device)

    def build_mlp(self, input_dim):
        layers = []
        hidden_dim = 1024
        for i in range(self.n_layers):
            out_dim = hidden_dim if i < self.n_layers - 1 else 128
            layers.append(nn.Linear(input_dim, out_dim))
            if i < self.n_layers - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
            input_dim = out_dim
        return nn.Sequential(*layers).to(device)

    def forward(self, u, i):
        return self.user_net(self.initial_user[u]), self.item_net(self.initial_item[i])

    def cal_cl_loss(self, i):
        i = torch.LongTensor(i).to(device)
        emb = self.initial_item[i]
        i1, i2 = self.dropout(emb), self.dropout(emb)
        return InfoNCE(self.item_net(i1), self.item_net(i2), self.tau)


class SSL4RecModel(GraphRecommender):
    def __init__(self, conf, train_data, test_data):
        super().__init__(conf, train_data, test_data)
        args = conf['SSL4Rec']
        self.cl_rate = args['alpha']
        self.tau = args['tau']
        self.drop = args['drop']
        self.n_layers = conf.get('n.layers', 1)
        self.reg_weight = conf.get('reg.weight', 0.0001)
        self.model = DNNEncoder(self.data, self.emb_size, self.drop, self.tau, self.n_layers)
        self.best_performance = {}

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        best_epoch, best_metric, patience, no_improv = 0, {}, 3, 0
        for epoch in range(self.maxEpoch):
            self.model.train()
            epoch_loss, batch_count = 0, 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                u, i, _ = batch
                u = torch.LongTensor(u).to(device)
                i = torch.LongTensor(i).to(device)
                u_emb, i_emb = self.model(u, i)
                rec_loss = batch_softmax_loss(u_emb, i_emb, self.tau)
                cl_loss = self.cl_rate * self.model.cal_cl_loss(i)
                batch_loss = rec_loss + cl_loss + l2_reg_loss(self.reg_weight, u_emb, i_emb)
                optimizer.zero_grad(); batch_loss.backward(); optimizer.step()
                epoch_loss += batch_loss.item()
                batch_count += 1
                if n % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {n}, Rec Loss: {rec_loss.item():.4f}, CL Loss: {cl_loss.item():.4f}")
            self.model.eval()
            with torch.no_grad():
                self.query_emb, self.item_emb = self.model(
                    torch.arange(self.data.user_num).to(device), 
                    torch.arange(self.data.item_num).to(device)
                )
            current = self.evaluate()
            if not best_metric or self.is_better(current, best_metric):
                best_metric, best_epoch = current, epoch
                self.save(); no_improv = 0
            else:
                no_improv += 1
            if no_improv >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        print(f"Best performance achieved at epoch {best_epoch+1}")
        return best_metric

    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, [self.topN])
        print(f"{'='*80}\nEvaluation:\n{''.join(metrics)}")
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.strip().split(':')]}

    def is_better(self, current, best):
        return current.get('Recall', 0) > best.get('Recall', 0)

    def save(self):
        with torch.no_grad():
            self.best_query_emb, self.best_item_emb = self.model(
                torch.arange(self.data.user_num).to(device), 
                torch.arange(self.data.item_num).to(device)
            )

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(self.query_emb[u], self.item_emb.T).cpu().numpy()


class Tuner:
    def __init__(self, train_set, test_set, base_conf):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_conf
        self.results = []
        self.grid = { 
            'n.layers': [1, 2, 3],
            'embedding.size': [32, 64, 128],
            'batch.size': [1024, 2048, 4096],
            'learning.rate': [0.001],
            'reg.lambda': [0.0001],
            'reg.weight': [0.0001, 0.001, 0.01],
            'SSL4Rec.tau': [0.07, 0.1, 0.2],
            'SSL4Rec.alpha': [0.1, 0.2, 0.3],
            'SSL4Rec.drop': [0.1, 0.2, 0.3]
        }

    def generate_param_combinations(self):
        keys = list(self.grid.keys())
        values = list(self.grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def run(self):
        combinations = self.generate_param_combinations()
        print(f"Starting hyperparameter tuning with {len(combinations)} combinations...")
        for i, params in enumerate(combinations, 1):
            conf = self.make_config(params)
            try:
                print(f"\n{'='*60}\nTraining combination {i}")
                print(f"Parameters: {conf['SSL4Rec']}")
                print(f"Embedding size: {conf['embedding.size']}, Batch size: {conf['batch.size']}")
                print(f"Learning rate: {conf['learning.rate']}, Reg lambda: {conf['reg.lambda']}")
                print('='*60)
                model = SSL4RecModel(conf, self.train_set, self.test_set)
                metrics = model.train()
                self.results.append({'config': conf, 'metrics': metrics})
                print(f"\nCompleted {i}/{len(combinations)} combinations")
                print(f"Best performance - NDCG: {metrics.get('NDCG')}, "
                      f"Recall: {metrics.get('Recall')}, "
                      f"Precision: {metrics.get('Precision')}, "
                      f"Hit Ratio: {metrics.get('Hit Ratio')}")
            except Exception as e:
                print(f"Error in config {i}: {e}")
                self.results.append({'config': conf, 'error': str(e)})

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'n.layers': params['n.layers'],
            'embedding.size': params['embedding.size'],
            'batch.size': params['batch.size'],
            'learning.rate': params['learning.rate'],
            'reg.lambda': params['reg.lambda'],
            'reg.weight': params['reg.weight'],
            'max.epoch': 1,
            'item.ranking.topN': [10, 20, 30, 50],
            'SSL4Rec': {
                'tau': params['SSL4Rec.tau'],
                'alpha': params['SSL4Rec.alpha'],
                'drop': params['SSL4Rec.drop'],
                'n.layers': params['n.layers'],
                'reg.weight': params['reg.weight']
            }
        })
        return conf

    def best(self, metric='Recall'):
        valid = [r for r in self.results if 'metrics' in r]
        return max(valid, key=lambda r: r['metrics'].get(metric, 0), default=None)

    def save(self, path='ssl4rec_results.json'):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved results to {path}")


# === Utility: Load data ===
def load_data(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [[*line.strip().split()[:2], 1.0] for line in f if line.strip()]


# === Utility: Print summary ===
def print_summary(results):
    successful = [r for r in results if 'metrics' in r]
    failed = [r for r in results if 'error' in r]
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING SUMMARY")
    print(f"Total combinations: {len(results)} | Successful: {len(successful)} | Failed: {len(failed)}")
    for metric in ['NDCG', 'Recall', 'Hit Ratio', 'Precision']:
        try:
            best = max(successful, key=lambda r: r['metrics'].get(metric, 0))
            print(f"Best {metric}: {best['metrics'][metric]} with config {best['config']['SSL4Rec']}")
        except:
            print(f"Metric {metric}: N/A")


# === Main runner ===
if __name__ == '__main__':
    print("Loading data from configuration files...")
    train_file = './dataset/ml100k/train.txt'
    test_file = './dataset/ml100k/test.txt'
    train_set = load_data(train_file)
    test_set = load_data(test_file)
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    base_conf = {
        'model': {'name': 'SSL4Rec'},
        'training.set': train_file,
        'test.set': test_file,
        'output': './results/',
    }
    print("SSL4Rec Hyperparameter Tuning Framework\n" + "="*50)
    print("This script will tune the following hyperparameters:")
    print("- Embedding size: [32, 64, 128]")
    print("- Batch size: [1024, 2048, 4096]")
    print("- Learning rate: [0.001]")
    print("- Regularization weight: [0.0001, 0.001, 0.01]")
    print("- Regularization lambda: [0.0001]")
    print("- SSL4Rec tau: [0.07, 0.1, 0.2]")
    print("- SSL4Rec alpha: [0.1, 0.2, 0.3]")
    print("- SSL4Rec drop: [0.1, 0.2, 0.3]")
    print(f"Total combinations: 243 = {3*3*1*1*3*3*3}\n")
    tuner = Tuner(train_set, test_set, base_conf)
    tuner.run()
    tuner.save()
    print_summary(tuner.results)