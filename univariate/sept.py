import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os, json, random, math, copy
from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------- Batching & Sampling ----------------------------
def next_batch_pairwise(data, batch_size):
    training_data = data.training_data
    random.shuffle(training_data)
    ptr = 0
    while ptr < len(training_data):
        batch_end = min(ptr + batch_size, len(training_data))
        users = [training_data[i][0] for i in range(ptr, batch_end)]
        items = [training_data[i][1] for i in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        for u, i in zip(users, items):
            u_idx.append(data.user[u])
            i_idx.append(data.item[i])
            while True:
                neg = random.choice(list(data.item.keys()))
                if neg not in data.training_set_u[u]:
                    j_idx.append(data.item[neg])
                    break
        yield torch.LongTensor(u_idx).to(device), torch.LongTensor(i_idx).to(device), torch.LongTensor(j_idx).to(device)


# ------------------------------ BPR Loss ----------------------------------
def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    return loss


# ---------------------- Sparse Adjacency Converter -----------------------
class TFGraphInterface:
    @staticmethod
    def convert_sparse_mat_to_tensor_inputs(X):
        coo = X.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)


# ---------------------- Graph Augmentor ------------------------------
class GraphAugmentor:
    @staticmethod
    def edge_dropout(sp_adj, drop_rate):
        row, col = sp_adj.nonzero()
        idx = np.arange(len(row))
        keep_idx = np.random.choice(idx, int(len(idx) * (1 - drop_rate)), replace=False)
        new_row, new_col = row[keep_idx], col[keep_idx]
        data = np.ones_like(new_row, dtype=np.float32)
        return sp.csr_matrix((data, (new_row, new_col)), shape=sp_adj.shape)


# ---------------------- Evaluation Metrics ---------------------------
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
            DCG = sum(1.0 / math.log2(i+2) for i, item in enumerate(res[u]) if item[0] in origin[u])
            IDCG = sum(1.0 / math.log2(i+2) for i in range(min(len(origin[u]), N)))
            score += DCG / IDCG if IDCG else 0
        return round(score / len(res), 5)


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


# ---------------------------- Data Interface ----------------------------
class Interaction:
    def __init__(self, conf, train, test):
        self.train, self.test = train, test
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
        self.user_num, self.item_num = len(self.user), len(self.item)
        self.training_data = [(u, i) for u, i, _ in self.train]
        self.training_set_u = {u: set() for u in self.user}
        for u, i, _ in self.train:
            self.training_set_u[u].add(i)
        self.test_set = {}
        for u, i, _ in self.test:
            self.test_set.setdefault(u, {})[i] = 1
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

    def get_user_id(self, user):
        return self.user[user]

    def user_rated(self, user):
        return list(self.training_set_u[user]), []


# ---------------------------- Base Recommender ----------------------------
class Recommender:
    def __init__(self, conf, train, test):
        self.config = conf
        self.model_name = conf['model']['name']
        self.output = conf.get('output', './')
        self.ranking = conf.get('item.ranking.topN', [10, 20, 30, 50])
        self.reOutput, self.result = [], []


class GraphRecommender(Recommender):
    def __init__(self, conf, train, test):
        super().__init__(conf, train, test)
        self.data = Interaction(conf, train, test)
        self.topN = self.ranking
        self.max_N = max(self.topN)
        self.bestPerformance = []

    def test(self):
        rec_list = {}
        for user in self.data.test_set:
            scores = self.predict(user)
            rated, _ = self.data.user_rated(user)
            for i in rated:
                scores[self.data.item[i]] = -1e8
            top_items = torch.topk(torch.tensor(scores), self.max_N)
            rec_list[user] = list(zip([self.data.id2item[i.item()] for i in top_items.indices], top_items.values.tolist()))
        return rec_list

    def evaluate(self, rec_list):
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print(f"Evaluation result:\n{''.join(self.result)}")
        return self.result

    def fast_evaluation(self, epoch):
        print(f"Evaluating Epoch {epoch+1}...")
        rec_list = self.test()
        performance = {k: float(v) for m in ranking_evaluation(self.data.test_set, rec_list, [self.max_N])[1:] for k, v in [m.strip().split(":")]}
        if not self.bestPerformance or performance.get("Recall", 0) > self.bestPerformance[1].get("Recall", 0):
            self.bestPerformance = [epoch+1, performance]
        return performance


# ---------------------------- SEPT Model (PyTorch) ----------------------------
class SEPT(GraphRecommender):
    def __init__(self, conf, train, test):
        super().__init__(conf, train, test)
        args = conf['SEPT']
        self.emb_size = conf.get('emb_size', 64)
        self.batch_size = conf.get('batch_size', 1024)
        self.lr = conf.get('lr', 0.001)
        self.reg_lambda = conf.get('reg_lambda', 1e-4)
        self.n_layers = int(args['n_layer'])
        self.drop_rate = float(args['drop_rate'])
        self.model = None
        self.build()

    def build(self):
        self.user_embeddings = nn.Embedding(self.data.user_num, self.emb_size).to(device)
        self.item_embeddings = nn.Embedding(self.data.item_num, self.emb_size).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.norm_adj = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(self.data.norm_adj)

    def parameters(self):
        return list(self.user_embeddings.parameters()) + list(self.item_embeddings.parameters())

    def encoder(self, emb, adj):
        all_embs = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            emb = F.normalize(emb, dim=1)
            all_embs.append(emb)
        return torch.stack(all_embs, dim=0).mean(0)

    def train(self):
        for epoch in range(self.config.get('max.epoch', 10)):
            self.user_embeddings.train()
            self.item_embeddings.train()
            all_emb = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
            dropped_adj = GraphAugmentor.edge_dropout(self.data.norm_adj, self.drop_rate)
            aug_adj = TFGraphInterface.convert_sparse_mat_to_tensor_inputs(dropped_adj)
            final_emb = self.encoder(all_emb, aug_adj)
            self.U, self.V = final_emb[:self.data.user_num], final_emb[self.data.user_num:]
            for n, (u_idx, i_idx, j_idx) in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                u_emb = self.U[u_idx]
                i_emb = self.V[i_idx]
                j_emb = self.V[j_idx]
                loss = bpr_loss(u_emb, i_emb, j_emb)
                reg_loss = self.reg_lambda * (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2) + j_emb.norm(2).pow(2)) / 2
                self.optimizer.zero_grad()
                (loss + reg_loss).backward(retain_graph=True)
                self.optimizer.step()
                if (n + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}| Batch: {n+1} Loss: {loss.item():.4f}")
            self.fast_evaluation(epoch)
        return self.evaluate()

    def predict(self, user):
        uid = self.data.get_user_id(user)
        return self.V @ self.U[uid].T
    
    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print("Detailed TopN Evaluation:")
        print("".join(metrics))
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':', 1)]}


# ---------------------------- Hyperparameter Tuner ----------------------------
class SEPTTuner:
    def __init__(self, train_set, test_set, base_config):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_config
        self.results = []
        self.grid = {
            'emb_size': [16, 32, 64, 128, 256, 512],
            'batch_size': [128, 256, 512, 1024, 2048, 4096],
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1], 
            'reg_lambda': [0.00001, 0.0001, 0.001, 0.01, 0.1], 
            'SEPT.n_layer': [1, 2, 3, 4],
            'SEPT.drop_rate': [0.1, 0.2, 0.3]
        }
        self.default = {
            'emb_size': 64, 
            'batch_size': 2048, 
            'lr': 0.001, 
            'reg_lambda': 0.0001,
            'SEPT.n_layer': 2, 
            'SEPT.drop_rate': 0.3
        }

    def run(self):
        total_runs = sum(len(v) for v in self.grid.values())
        print(f"\nTotal combinations: {total_runs}\n" + '='*80)
        run_count = 0
        for key, values in self.grid.items():
            print(f"\n{'='*80}\n Tuning hyperparameter: {key}")
            for val in values:
                run_count += 1
                print(f"\n>>> [{run_count}/{total_runs}] Tuning {key} = {val}")
                param = copy.deepcopy(self.default)
                param[key] = val
                conf = self.make_config(param)
                try:
                    model = SEPT(conf, self.train_set, self.test_set)
                    metrics = model.train()
                    print(f"Result: NDCG: {metrics.get('NDCG')}, Recall: {metrics.get('Recall')}, "
                          f"Precision: {metrics.get('Precision')}, Hit Ratio: {metrics.get('Hit Ratio')}")
                    self.results.append({'config': conf, 'metrics': metrics})
                except Exception as e:
                    import traceback
                    print(f"Error tuning {key} = {val}: {e}")
                    traceback.print_exc()
        with open('sept_tuning_individual.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'emb_size': params['emb_size'],
            'batch_size': params['batch_size'],
            'lr': params['lr'],
            'reg_lambda': params['reg_lambda'],
            'max.epoch': 10,
            'item.ranking.topN': [10, 20, 30, 50],
            'SEPT': {
                'n_layer': params['SEPT.n_layer'],
                'drop_rate': params['SEPT.drop_rate']
            }
        })
        return conf


# ---------------------------- Load Data ----------------------------
def load_data(path):
    return [[*line.strip().split()[:2], 1.0] for line in open(path) if line.strip()] if os.path.exists(path) else []


# ---------------------------- Main ----------------------------
if __name__ == '__main__':
    print("Loading data from configuration files...")
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'SEPT', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print("\nSEPT Hyperparameter Tuning Framework\n" + "="*80)
    tuner = SEPTTuner(train_set, test_set, base_config)
    tuner.run()
