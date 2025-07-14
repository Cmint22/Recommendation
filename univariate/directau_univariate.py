import os, json, copy, math, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import shuffle, choice
from datetime import datetime
from os.path import abspath
from time import strftime, localtime, time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# === Pairwise Negative Sampler ===
def next_batch_pairwise(data, batch_size, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    while ptr < len(training_data):
        batch_end = min(ptr + batch_size, len(training_data))
        users = [training_data[i][0] for i in range(ptr, batch_end)]
        items = [training_data[i][1] for i in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        for user, item in zip(users, items):
            u_idx.append(data.user[user])
            i_idx.append(data.item[item])
            while True:
                neg_item = choice(list(data.item.keys()))
                if neg_item not in data.training_set_u[user]:
                    j_idx.append(data.item[neg_item])
                    break
        yield u_idx, i_idx, j_idx

# === L2 Regularization Loss ===
def l2_reg_loss(reg, *args):
    return reg * sum(torch.norm(x, p=2) / x.shape[0] for x in args)

# === Evaluation Metrics ===
class Metric:
    @staticmethod
    def hits(origin, res):
        return {u: len(set(origin[u]).intersection(i[0] for i in res[u])) for u in origin}
    
    @staticmethod
    def hit_ratio(origin, hits):
        total = sum(len(origin[u]) for u in origin)
        return round(sum(hits.values()) / total, 5)

    @staticmethod
    def precision(hits, N):
        return round(sum(hits.values()) / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        return round(np.mean([hits[u] / len(origin[u]) for u in hits]), 5)

    @staticmethod
    def NDCG(origin, res, N):
        score = 0
        for u in res:
            DCG = sum(1.0 / math.log2(i + 2) for i, item in enumerate(res[u]) if item[0] in origin[u])
            IDCG = sum(1.0 / math.log2(i + 2) for i in range(min(len(origin[u]), N)))
            score += DCG / IDCG if IDCG else 0
        return round(score / len(res), 5)

# === Ranking Evaluation Wrapper ===
def ranking_evaluation(origin, res, N):
    results = []
    for n in N:
        pred = {u: res[u][:n] for u in res}
        hits = Metric.hits(origin, pred)
        results.append(f'Top {n}\n')
        results += [
            f'Hit Ratio:{Metric.hit_ratio(origin, hits)}\n',
            f'Precision:{Metric.precision(hits, n)}\n',
            f'Recall:{Metric.recall(hits, origin)}\n',
            f'NDCG:{Metric.NDCG(origin, pred, n)}\n'
        ]
    return results

# === Torch Sparse Matrix Interface ===
class TorchGraphInterface:
    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape).to(device)

# === Minimal DataLoader and File I/O ===
def load_data(path):
    return [[*line.strip().split()[:2], 1.0] for line in open(path) if line.strip()] if os.path.exists(path) else []

class FileIO:
    @staticmethod
    def write_file(path, filename, content):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), 'w') as f:
            f.writelines(content if isinstance(content, list) else [content])

# === Interaction structure ===
class Interaction:
    def __init__(self, conf, train, test):
        self.train = train
        self.test = test
        self.user, self.item = {}, {}
        self.id2user, self.id2item = {}, {}
        self.training_data, self.test_set = [], {}
        self._build()

    def _build(self):
        users, items = set(), set()
        for u, i, _ in self.train:
            users.add(u)
            items.add(i)
        self.user = {u: idx for idx, u in enumerate(sorted(users))}
        self.item = {i: idx for idx, i in enumerate(sorted(items))}
        self.id2user = {idx: u for u, idx in self.user.items()}
        self.id2item = {idx: i for i, idx in self.item.items()}
        self.user_num = len(self.user)
        self.item_num = len(self.item)
        self.training_data = [(u, i) for u, i, _ in self.train]
        self.training_set_u = {u: set() for u in self.user}
        for u, i, _ in self.train:
            self.training_set_u[u].add(i)
        self.test_set = {}
        for u, i, _ in self.test:
            self.test_set.setdefault(u, {})
            self.test_set[u][i] = 1
        self.norm_adj = self._build_adj()

    def _build_adj(self):
        from scipy.sparse import coo_matrix
        num_nodes = self.user_num + self.item_num
        row, col, data = [], [], []
        for u, i, _ in self.train:
            uid, iid = self.user[u], self.item[i] + self.user_num
            row += [uid, iid]
            col += [iid, uid]
            data += [1, 1]
        return coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    def get_user_id(self, user): return self.user[user]
    def user_rated(self, user): return list(self.training_set_u[user]), []

# === Recommender Base Class ===
class Recommender:
    def __init__(self, conf, train_set, test_set, **kwargs):
        self.config = conf
        self.output = conf.get('output', './')
        self.model_name = conf['model']['name']
        self.ranking = conf.get('item.ranking.topN', [10, 20, 30, 50])
        self.recOutput, self.result, self.model_log = [], [], []

    def print_model_info(self): print(f'Using model {self.model_name}')
    def save(self): pass  # Implement save logic if needed

# === GraphRecommender Base ===
class GraphRecommender(Recommender):
    def __init__(self, conf, train_set, test_set, **kwargs):
        super().__init__(conf, train_set, test_set, **kwargs)
        self.data = Interaction(conf, train_set, test_set)
        self.topN = [int(n) for n in self.ranking]
        self.max_N = max(self.topN)
        self.bestPerformance = []

    def test(self):
        rec_list = {}
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            rated_list, _ = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -1e8
            top_items = torch.topk(torch.tensor(candidates, device=device), self.max_N)
            item_names = [self.data.id2item[idx] for idx in top_items.indices.tolist()]
            scores = top_items.values.tolist()
            rec_list[user] = list(zip(item_names, scores))
        return rec_list

    def evaluate(self, rec_list):
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print(f"Evaluation result:\n{''.join(self.result)}")
        return self.result

    def fast_evaluation(self, epoch):
        print(f'Evaluating Epoch {epoch+1}...')
        rec_list = self.test()
        performance = {k: float(v) for m in ranking_evaluation(self.data.test_set, rec_list, [self.max_N])[1:] for k, v in [m.strip().split(":")]}
        if not self.bestPerformance or performance.get("Recall", 0) > self.bestPerformance[1].get("Recall", 0):
            self.bestPerformance = [epoch+1, performance]
        return performance


# === DirectAU Model ===
class DirectAU(GraphRecommender):
    def __init__(self, conf, train_set, test_set):
        super().__init__(conf, train_set, test_set)
        args = self.config['DirectAU']
        self.gamma = float(args['gamma'])
        self.n_layers = int(args['n_layers'])
        self.reg = conf.get("reg.lambda", 0.0001)
        self.batch_size = conf.get("batch.size", 512)
        self.emb_size = conf.get("embedding.size", 64)
        self.lRate = conf.get("learning.rate", 0.001)
        self.optimizer = conf.get("optimizer", "adam").lower()
        self.model = LGCNEncoder(self.data, self.emb_size, self.n_layers).to(device)

    def train(self):
        opt_type = self.config.get("optimizer", "adam").lower()
        if opt_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lRate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")
        for epoch in range(1):
            for n, (user_idx, pos_idx, neg_idx) in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                self.model.train()
                user_emb, item_emb, _ = self.model()
                user_emb, item_emb = user_emb.to(device), item_emb.to(device)
                u_emb, p_emb, n_emb = user_emb[user_idx].to(device), item_emb[pos_idx].to(device), item_emb[neg_idx].to(device)
                pos_loss = self.calculate_loss(u_emb, p_emb)
                neg_loss = self.calculate_loss(u_emb, n_emb)
                loss = pos_loss - neg_loss
                loss += l2_reg_loss(self.reg, u_emb, p_emb, n_emb) / self.batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (n + 1) % 100 == 0:
                    align = self.alignment(u_emb, p_emb).item()
                    uniform = self.gamma * (self.uniformity(u_emb).item() + self.uniformity(p_emb).item()) / 2
                    print(f"Batch {n+1}, Align loss: {align:.4f}, Uniform loss: {uniform:.4f}, L2 regulation: {l2_reg_loss(self.reg, u_emb, p_emb, n_emb).item():.4f}, Total loss: {loss.item():.4f}")
        with torch.no_grad():
            self.user_emb, self.item_emb, _ = self.model()
            self.user_emb = self.user_emb.detach().to(device)
            self.item_emb = self.item_emb.detach().to(device)
        return self.evaluate()

    def calculate_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = self.gamma * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align + uniform

    def alignment(self, x, y):
        return (F.normalize(x, dim=-1) - F.normalize(y, dim=-1)).pow(2).sum(1).mean()

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        pdist = torch.pdist(x, p=2)
        return (pdist.pow(2).mul(-t).exp().mean() + 1e-8).log() if pdist.numel() > 0 else torch.tensor(0.0, device=device)

    def predict(self, u):
        u = self.data.get_user_id(u)
        if not hasattr(self, 'user_emb'):
            self.user_emb, self.item_emb, _ = self.model()
            self.user_emb = self.user_emb.detach().to(device)
            self.item_emb = self.item_emb.detach().to(device)
        return torch.matmul(self.user_emb[u], self.item_emb.T).cpu().numpy()

    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print("Detailed TopN Evaluation:")
        print("".join(metrics))
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':', 1)]}

# === LightGCN Encoder ===
class LGCNEncoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super().__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        return nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size, device=device))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size, device=device)))
        })

    def forward(self):
        emb = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_emb = [emb]
        for _ in range(self.layers):
            emb = torch.sparse.mm(self.sparse_norm_adj, emb)
            all_emb.append(emb)
        final_emb = torch.stack(all_emb).mean(0)
        return final_emb[:self.data.user_num], final_emb[self.data.user_num:], all_emb


# === Hyperparameter Tuner ===
class DirectAUTuner:
    def __init__(self, train_set, test_set, base_config):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_config
        self.results = []
        self.grid = {
            'embedding.size': [16, 32, 64, 128, 256, 512],
            'batch.size': [256, 512, 1024, 2048, 4096],
            'learning.rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            'reg.lambda': [1e-4, 1e-3, 1e-2],
            'optimizer': ['adam', 'sgd'],
            'DirectAU.gamma': [0.5, 1.0, 3.0],
            'DirectAU.n_layers': [1, 2, 3, 4, 5, 6]
        }
        self.default = {
            'embedding.size': 64,
            'batch.size': 2048,
            'learning.rate': 1e-3,
            'reg.lambda': 1e-4,
            'optimizer': 'adam',
            'DirectAU.gamma': 1.0,
            'DirectAU.n_layers': 3
        }

    def run(self):
        total_runs = sum(len(v) for v in self.grid.values())
        print(f"\nTotal combinations: {total_runs}\n" + '='*80)
        run_count = 0
        for key, values in self.grid.items():
            print(f"\n{'='*80}\nTuning hyperparameter: {key}")
            for val in values:
                run_count += 1
                print(f"\n>>> [{run_count}/{total_runs}] {key} = {val}")
                param_config = self.default.copy()
                param_config[key] = val
                conf = self.make_config(param_config)
                try:
                    # print(f"\n>>> {key} = {val}")
                    model = DirectAU(conf, self.train_set, self.test_set)
                    metrics = model.train()
                    print(f"Result: NDCG: {metrics.get('NDCG')}, Recall: {metrics.get('Recall')}, "
                          f"Precision: {metrics.get('Precision')}, Hit Ratio: {metrics.get('Hit Ratio')}")
                    self.results.append({'config': conf, 'metrics': metrics})
                except Exception as e:
                    import traceback
                    print(f"Error tuning {key} = {val}: {e}")
                    traceback.print_exc()
        with open('directau_tuning_individual.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'embedding.size': params['embedding.size'],
            'batch.size': params['batch.size'],
            'learning.rate': params['learning.rate'],
            'reg.lambda': params['reg.lambda'],
            'optimizer': params['optimizer'],
            'max.epoch': 1,
            'item.ranking.topN': [10, 20, 30, 50],
            'DirectAU': {
                'gamma': params['DirectAU.gamma'],
                'n_layers': params['DirectAU.n_layers']
            }
        })
        for key in ['gamma', 'n_layers']:
            conf[f'DirectAU.{key}'] = conf['DirectAU'][key]
        return conf


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
                  f"embedding.size={conf.get('embedding.size')}, "
                  f"batch.size={conf.get('batch.size')}, "
                  f"learning.rate={conf.get('learning.rate')}, "
                  f"reg.lambda={conf.get('reg.lambda')}, "
                  f"optimizer={conf.get('optimizer')}, "
                  f"gamma={conf['DirectAU']['gamma']}, "
                  f"n_layers={conf['DirectAU']['n_layers']}")
        except Exception as e:
            print(f"[Best {metric}] Error: {e}")

# === Entry Point ===
if __name__ == '__main__':
    print("Loading data from configuration files...")
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'DirectAU', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print("\nDirectAU Hyperparameter Tuning Framework\n" + "="*80)
    tuner = DirectAUTuner(train_set, test_set, base_config)
    tuner.run()
    # print_summary(tuner.results)