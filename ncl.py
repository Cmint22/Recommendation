import os, json, copy, math, itertools, heapq
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, scipy.sparse as sp
from datetime import datetime
from random import shuffle, choice
from collections import defaultdict
from time import strftime, localtime, time
from numba import jit
from os.path import abspath
from re import split
import logging
from os import remove
import faiss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================== Data Class ====================
class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training
        self.test_data = test

# =================== Interaction Class (inherits Data, Graph) ====================
class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

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

# =================== Loss Functions ====================
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
            neg_trials = 0
            while True:
                neg_item = choice(list(data.item.keys()))
                if neg_item not in data.training_set_u[user]:
                    j_idx.append(data.item[neg_item])
                    break
                neg_trials += 1
                if neg_trials > 100:  # tránh lặp vô hạn
                    break
        if len(u_idx) == len(i_idx) == len(j_idx) and u_idx:
            yield u_idx, i_idx, j_idx

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    return reg * sum(torch.norm(x, p=2) / x.shape[0] for x in args)

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

# =================== Metrics & Evaluation ====================
class Metric:
    @staticmethod
    @staticmethod
    def hits(origin, res):
        return {
            u: len(set(origin[u]).intersection(i[0] for i in res.get(u, [])))
            for u in origin if u in res
        }
    
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

# =================== Utility Tools ====================
@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    k_largest_scores = [item[0] for item in n_candidates]
    return ids, k_largest_scores

def normalize(vec, maxVal, minVal):
    if maxVal > minVal:
        return (vec - minVal) / (maxVal - minVal)
    elif maxVal == minVal:
        return vec / maxVal
    else:
        print('error... maximum value is less than minimum value.')
        raise ArithmeticError

class TorchGraphInterface:
    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape).to(device)

# =================== Logging, I/O ====================
class Log(object):
    def __init__(self,module,filename):
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        handler = logging.FileHandler('./log/'+filename+'.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self,text):
        self.logger.info(text)

class FileIO:
    @staticmethod
    def write_file(path, filename, content):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), 'w') as f:
            f.writelines(content if isinstance(content, list) else [content])

# =================== Recommender Base ====================
class Recommender:
    def __init__(self, conf, train_set, test_set, **kwargs):
        self.config = conf
        self.output = conf.get('output', './')
        self.model_name = conf['model']['name']
        self.ranking = conf.get('item.ranking.topN', [10, 20, 30, 50])
        self.recOutput, self.result, self.model_log = [], [], []
    def print_model_info(self): print(f'Using model {self.model_name}')
    def save(self): pass


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



# =================== NCL Model ====================
class NCLModel(GraphRecommender):
    def __init__(self, conf, train_set, test_set):
        super(NCLModel, self).__init__(conf, train_set, test_set)
        args = self.config['NCL']
        self.n_layers = args['n_layers']
        self.ssl_temp = args['tau']
        self.ssl_reg = args['ssl_reg']
        self.proto_reg = args['proto_reg']
        self.hyper_layers = args['hyper_layers']
        self.alpha = args['alpha']
        self.k = args['num_clusters']
        self.batch_size = conf.get('batch.size', 2048)
        self.emb_size = conf.get('embedding.size', 64)
        self.lRate = conf.get('learning.rate', 0.001)
        self.reg = conf.get('reg.lambda', 0.0001)
        self.model = LGCNEncoder(self.data, self.emb_size, self.n_layers)
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def train(self):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        self.model.train()
        for epoch in range(1):
            if epoch >= 0:
                self.e_step()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb, rec_item_emb, emb_list = model()
                user_emb = rec_user_emb[user_idx].to(device)
                pos_emb = rec_item_emb[pos_idx].to(device)
                neg_emb = rec_item_emb[neg_idx].to(device)
                rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)
                initial_emb = emb_list[0]
                if self.hyper_layers * 2 >= len(emb_list):
                    context_emb = emb_list[-1]
                else:
                    context_emb = emb_list[self.hyper_layers * 2]
                ssl_loss = self.ssl_layer_loss(context_emb, initial_emb, user_idx, pos_idx)
                self.e_step()
                proto_loss = self.ProtoNCE_loss(initial_emb, user_idx, pos_idx)
                total_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_emb, neg_emb) / self.batch_size + ssl_loss + proto_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if (n + 1) % 100 == 0:
                    print(f"Batch {n+1}: Rec_loss={rec_loss.item():.4f}, ssl_loss={ssl_loss.item():.4f}, Proto_loss={proto_loss.item():.4f}, Total_loss={total_loss.item():.4f}")
        self.model.eval()
        with torch.no_grad():
            self.user_emb, self.item_emb, _ = self.model()
        rec_list = self.test()
        if rec_list is None:
            raise ValueError("rec_list is None. Please check your test() implementation.")
        return self.evaluate()

    def e_step(self):
        user_emb, item_emb, _ = self.model()
        user_np = user_emb.detach().cpu().numpy()
        item_np = item_emb.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_np)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_np)

    def run_kmeans(self, x):
        # if x.shape[0] < self.k * 39:
        #     self.k = min(self.k, max(10, x.shape[0] // 10))
        max_k = max(2, x.shape[0] // 39)  # ít nhất 2 cụm
        self.k = min(self.k, max_k)
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=False)
        kmeans.train(x)
        centroids = torch.tensor(kmeans.centroids).to(device)
        _, I = kmeans.index.search(x, 1)
        return centroids, torch.LongTensor(I).squeeze()

    def ssl_layer_loss(self, context, initial, user, item):
        cu, ci = context[:self.data.user_num], context[self.data.user_num:]
        iu, ii = initial[:self.data.user_num], initial[self.data.user_num:]
        norm_cu, norm_iu = F.normalize(cu[user]), F.normalize(iu[user])
        norm_ci, norm_ii = F.normalize(ci[item]), F.normalize(ii[item])
        ssl_loss_u = -torch.log(torch.exp((norm_cu * norm_iu).sum(1)/self.ssl_temp) /
                         torch.exp(torch.matmul(norm_cu, F.normalize(iu).T)/self.ssl_temp).sum(1)).sum()
        ssl_loss_i = -torch.log(torch.exp((norm_ci * norm_ii).sum(1)/self.ssl_temp) /
                         torch.exp(torch.matmul(norm_ci, F.normalize(ii).T)/self.ssl_temp).sum(1)).sum()
        return self.ssl_reg * (ssl_loss_u + self.alpha * ssl_loss_i)

    def ProtoNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        user2centroids = self.user_centroids[self.user_2cluster[user_idx]].to(device)
        item2centroids = self.item_centroids[self.item_2cluster[item_idx]].to(device)
        loss_user = InfoNCE(user_emb[user_idx], user2centroids, self.ssl_temp) * self.batch_size
        loss_item = InfoNCE(item_emb[item_idx], item2centroids, self.ssl_temp) * self.batch_size
        return self.proto_reg * (loss_user + loss_item)

    def evaluate(self):
        rec_list = self.test()
        if rec_list is None:
            raise ValueError("rec_list is None. Check if test() returns predictions.")
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print("Detailed TopN Evaluation :")
        print("".join(metrics))
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':', 1)]}

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        if not hasattr(self, 'user_emb') or not hasattr(self, 'item_emb'):
            self.user_emb, self.item_emb, _ = self.model()
        return torch.matmul(self.user_emb[u], self.item_emb.T).cpu().numpy()


class LGCNEncoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super().__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model().to(device)
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cpu()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        emb = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0).to(device)
        all_emb = [emb]
        for _ in range(self.layers):
            emb = torch.sparse.mm(self.sparse_norm_adj, emb)
            all_emb.append(emb)
        final_emb = torch.stack(all_emb).mean(0)
        return final_emb[:self.data.user_num], final_emb[self.data.user_num:], all_emb


class Tuner:
    def __init__(self, train_set, test_set, base_conf):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_conf
        self.results = []
        # self.grid = {
        #     'embedding.size': [64],
        #     'batch.size': [2048],
        #     'learning.rate': [0.001],
        #     'reg.lambda': [0.0001],
        #     'max.epoch': [1],
        #     'NCL.n_layers': [3],
        #     'NCL.tau': [0.1],
        #     'NCL.ssl_reg': [0.0001],
        #     'NCL.proto_reg': [0.0001],
        #     'NCL.alpha': [1.0],
        #     'NCL.num_clusters': [1000],
        #     'NCL.hyper_layers': [2]
        # }
        self.grid = {
            'embedding.size': [32, 64, 128, 256, 512, 1024],
            'batch.size': [64, 128, 256, 512, 1024, 2048, 4096],
            'learning.rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
            'reg.lambda': [1e-4, 5e-4, 1e-3],
            'max.epoch': [1],
            'NCL.n_layers': [1, 2, 3, 4, 5],
            'NCL.tau': [0.1, 0.2, 0.3],
            'NCL.ssl_reg': [1e-5, 1e-4, 1e-3],
            'NCL.proto_reg': [1e-5, 1e-4, 1e-3],
            'NCL.alpha': [0.3, 0.5, 0.6],
            'NCL.num_clusters': [20, 30, 50, 100, 200, 300],
            'NCL.hyper_layers': [2, 3, 4]
        }


    def run(self):
        keys, vals = list(self.grid.keys()), list(self.grid.values())
        combinations = list(itertools.product(*vals))
        print(f"\nNCL Hyperparameter Tuning - Total combinations: {len(combinations)}")
        best_result = None
        for i, combo in enumerate(combinations, 1):
            conf = self.make_config(dict(zip(keys, combo)))
            try:
                print(f"\n{'='*80}\n[{i}] Training with configuration:")
                print(f"Embedding size: {conf['embedding.size']}, Batch size: {conf['batch.size']}, Learning rate: {conf['learning.rate']}")
                print(f"Reg lambda: {conf['reg.lambda']}, NCL layers: {conf['NCL.n_layers']}, Tau: {conf['NCL.tau']}, SSL Reg: {conf['NCL.ssl_reg']}")
                print(f"Proto Reg: {conf['NCL.proto_reg']}, Alpha: {conf['NCL.alpha']}, Num clusters: {conf['NCL.num_clusters']}, Hyper layers: {conf['NCL.hyper_layers']}")
                model = NCLModel(conf, self.train_set, self.test_set)
                metrics = model.train()
                print(f"\nCompleted {i}/{len(combinations)} combinations")
                print(f"best performance - NDCG: {metrics.get('NDCG'):.5f}, "
                      f"Recall: {metrics.get('Recall'):.5f}, "
                      f"Precision: {metrics.get('Precision'):.5f}, "
                      f"Hit Ratio: {metrics.get('Hit Ratio'):.5f}")
                self.results.append({'config': conf, 'metrics': metrics})
                result = {'config': conf, 'metrics': metrics}
                self.results.append(result)
                if best_result is None or metrics.get('Recall', 0) > best_result['metrics'].get('Recall', 0):
                    best_result = result
            except Exception as e:
                import traceback
                print(f"Error in config {i}: {e}")
                traceback.print_exc()
                self.results.append({'config': conf, 'error': str(e)})

        if best_result:
            with open('ncl_results.json', 'w') as f:
                json.dump([best_result], f, indent=2)
            print("Saved best result to ncl_results.json")

    def make_config(self, params):
        required_keys = [
            'embedding.size', 'batch.size', 'learning.rate', 'reg.lambda', 'max.epoch',
            'NCL.n_layers', 'NCL.tau', 'NCL.ssl_reg', 'NCL.proto_reg',
            'NCL.alpha', 'NCL.num_clusters', 'NCL.hyper_layers'
        ]
        
        # Kiểm tra đủ số lượng keys trong params
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required hyperparameter: {key}")
        conf = copy.deepcopy(self.base)
        conf.update({
            'embedding.size': params['embedding.size'],
            'batch.size': params['batch.size'],
            'learning.rate': params['learning.rate'],
            'reg.lambda': params['reg.lambda'],
            'max.epoch': params['max.epoch'],
            'item.ranking.topN': [10, 20, 30, 50],
            'NCL': {
                'n_layers': params['NCL.n_layers'],
                'tau': params['NCL.tau'],
                'ssl_reg': params['NCL.ssl_reg'],
                'proto_reg': params['NCL.proto_reg'],
                'alpha': params['NCL.alpha'],
                'num_clusters': params['NCL.num_clusters'],
                'hyper_layers': params['NCL.hyper_layers']
            }
        })
        for key in ['n_layers', 'tau', 'ssl_reg', 'proto_reg', 'alpha', 'num_clusters', 'hyper_layers']:
            conf[f'NCL.{key}'] = conf['NCL'][key]
        return conf

    def best(self, metric='Recall'):
        valid = [r for r in self.results if 'metrics' in r]
        return max(valid, key=lambda r: r['metrics'].get(metric, 0), default=None)

    def save(self, path='ncl_results.json'):
        # best = self.best()
        # if best:
            with open(path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Saved best result to {path}")
        # else:
        #     print("No valid result to save.")


def load_data(path):
    return [[*line.strip().split()[:2], 1.0] for line in open(path) if line.strip()] if os.path.exists(path) else []


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
                  f"n_layers={conf['NCL']['n_layers']}, "
                  f"tau={conf['NCL']['tau']}, "
                  f"ssl_reg={conf['NCL']['ssl_reg']}, "
                  f"proto_reg={conf['NCL']['proto_reg']}, "
                  f"alpha={conf['NCL']['alpha']}, "
                  f"num_clusters={conf['NCL']['num_clusters']}, "
                  f"hyper_layers={conf['NCL']['hyper_layers']}")
        except Exception as e:
            print(f"[Best {metric}] Error: {e}")


if __name__ == '__main__':
    print("Loading data from configuration files...")
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'NCL', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print("\nNCL Hyperparameter Tuning Framework\n" + "="*80)
    print("This script will tune the following hyperparameters:")
    print("- Embedding size: [32, 64, 128, 256, 512, 1024]")
    print("- Batch size: [64, 128, 256, 512, 1024, 2048, 4096]")
    print("- Learning rate: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]")
    print("- Regularization lambda: [1e-4, 5e-4, 1e-3]")
    print("- Max epochs: [1]")
    print("- NCL layers: [1, 2, 3, 4, 5]")
    print("- NCL tau: [0.1, 0.2, 0.3]")
    print("- NCL SSL regularization: [1e-5, 1e-4, 1e-3]")
    print("- NCL proto regularization: [1e-5, 1e-4, 1e-3]")
    print("- NCL alpha: [0.3, 0.5, 0.6]")
    print("- NCL num clusters: [20, 30, 50, 100, 200, 300]")
    print("- NCL hyper layers: [2, 3, 4]")
    # print(f"Total combinations: 3149280 = {4*6*6*3*5*3*3*3*3*6*3}\n")
    tuner = Tuner(train_set, test_set, base_config)
    tuner.run()
    tuner.save()
    print_summary(tuner.results)