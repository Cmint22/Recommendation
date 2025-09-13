# !pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric 
# !pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu118

# !pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html 
# !pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html 
# !pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html 
# !pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html 
# !pip install torch-geometric 

import os, json, math, copy, torch, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
from torch.optim import Adam
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load data từ file txt: mỗi dòng "user item"
def load_data(path):
    return [[*line.strip().split()[:2], 1.0] for line in open(path) if line.strip()] if os.path.exists(path) else []


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


def build_movielens_graph(interaction: Interaction):
  adj = interaction.norm_adj.tocoo()
  edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
  num_nodes = interaction.user_num + interaction.item_num
  x = torch.eye(num_nodes, dtype=torch.float)
  return Data(x=x, edge_index=edge_index)


class Recommender:
    def __init__(self, conf, train_set, test_set, **kwwrgs):
        self.config = conf
        self.output = conf.get('output', './')
        self.model_name = conf['model']['name']
        self.ranking = conf.get('item.ranking.topN', [10, 20, 30, 50])
        self.reOutput, self.result, self.model_log = [], [], []

    def print_model_info(self): print(f'Using model {self.model_name}')
    def save(self): pass


class GraphRecommender(Recommender):
    def __init__(self, conf, train_set, test_set, encoder=None, **kwargs):
        super().__init__(conf, train_set, test_set, **kwargs)
        self.data = Interaction(conf, train_set, test_set)
        self.topN = [int(n) for n in self.ranking]
        self.max_N = max(self.topN)
        self.bestPerformance = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.graph = build_movielens_graph(self.data).to(self.device)

    def predict(self, user):
        if self.encoder is None:
          num_items = len(self.data.item)
          return np.random.rand(num_items)
        self.encoder.eval()
        with torch.no_grad():
          z, _ = self.encoder.online_encoder(self.graph.x, self.graph.edge_index)
        user_id = self.data.get_user_id(user)
        user_emb = z[user_id]
        item_embs = z[self.data.user_num:]
        scores = torch.matmul(item_embs, user_emb)
        return scores.cpu().tolist()

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

    def fast_evaluation(self, epoch, topK=None):
        print(f'Evaluating Epoch {epoch+1}...')
        rec_list = self.test()
        max_N = max(topK) if topK is not None else self.max_N
        performance = {k: float(v) for m in ranking_evaluation(self.data.test_set, rec_list, [self.max_N])[1:] for k, v in [m.strip().split(":")]}
        if not self.bestPerformance or performance.get("Recall", 0) > self.bestPerformance[1].get("Recall", 0):
            self.bestPerformance = [ epoch+1, performance]
        return performance


class Sampler(ABC):
    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)
        return ret

    @abstractmethod
    def sample(self, anchor, sample, *args, **kwargs):
        pass

    @staticmethod
    def add_intraview_negs(anchor, sample, pos_mask, neg_mask):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)                     # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)     # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)     # M * (M+N)
        return anchor, new_sample, new_pos_mask, new_neg_mask


class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask


class CrossScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(CrossScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, batch=None, neg_sample=None, use_gpu=True, *args, **kwargs):
        num_graphs = anchor.shape[0]  # M
        num_nodes = sample.shape[0]   # N
        device = sample.device
        if neg_sample is not None:
            assert num_graphs == 1  # only one graph, explicit negative samples are needed
            assert sample.shape == neg_sample.shape
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)     # M * 2N
            sample = torch.cat([sample, neg_sample], dim=0)         # 2N * K
        else:
            assert batch is not None
            if use_gpu:
                ones = torch.eye(num_nodes, dtype=torch.float32, device=device)     # N * N
                pos_mask = scatter(ones, batch, dim=0, reduce='sum')                # M * N
            else:
                pos_mask = torch.zeros((num_graphs, num_nodes), dtype=torch.float32).to(device)
                for node_idx, graph_idx in enumerate(batch):
                    pos_mask[graph_idx][node_idx] = 1.                              # M * N
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask

def get_sampler(mode: str, intraview_negs: bool) -> Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler(intraview_negs=intraview_negs)
    elif mode == 'G2L':
        return CrossScaleSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'unsupported mode: {mode}')

def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

class BootstrapContrast(torch.nn.Module):
    def __init__(self, loss, mode='L2L'):
        super(BootstrapContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=False)

    def forward(self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None,
                g1_pred=None, g2_pred=None, g1_target=None, g2_target=None,
                batch=None, extra_pos_mask=None):
        if self.mode == 'L2L':
            assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=h1_target, sample=h2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=h2_target, sample=h1_pred)
        elif self.mode == 'G2G':
            assert all(v is not None for v in [g1_pred, g2_pred, g1_target, g2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=g2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=g1_pred)
        else:
            assert all(v is not None for v in [h1_pred, h2_pred, g1_target, g2_target])
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                pos_mask1 = pos_mask2 = torch.ones([1, h1_pred.shape[0]], device=h1_pred.device)
                anchor1, sample1 = g1_target, h2_pred
                anchor2, sample2 = g2_target, h1_pred
            else:
                anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=h2_pred, batch=batch)
                anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=h1_pred, batch=batch)
        pos_mask1, _ = add_extra_mask(pos_mask1, extra_pos_mask=extra_pos_mask)
        pos_mask2, _ = add_extra_mask(pos_mask2, extra_pos_mask=extra_pos_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2)
        return (l1 + l2) * 0.5

def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }

def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]

def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split
        result = self.evaluate(x, y, split)
        return result


class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')
        return {
            'micro_f1': test_micro,
            'macro_f1': test_macro,
        }

class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None):
        if linear:
            self.evaluator = LinearSVC()
        else:
            self.evaluator = SVC()
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, params)

class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()

class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g

def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0
    return x

class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss

class BootstrapLatent(Loss):
    def __init__(self):
        super(BootstrapLatent, self).__init__()

    def compute(self, anchor, sample, pos_mask, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample = F.normalize(sample, dim=-1, p=2)

        similarity = anchor @ sample.t()
        loss = (similarity * pos_mask).sum(dim=-1)
        return loss.mean()

class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

# Chuyển từ list interactions sang PyG Data
def build_graph_from_interactions(edge_list, hidden_dim):
    from collections import defaultdict
    user2id, item2id = {}, {}
    uid, iid = 0, 0
    edges = []
    for u, i, _ in edge_list:
        if u not in user2id:
            user2id[u] = uid
            uid += 1
        if i not in item2id:
            item2id[i] = iid
            iid += 1
        edges.append((user2id[u], len(user2id) + item2id[i]))  # bipartite graph
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = len(user2id) + len(item2id)
    # x = torch.eye(num_nodes, dtype=torch.float)
    emb_layer = nn.Embedding(num_nodes, hidden_dim)
    x = emb_layer.weight  # shape = (num_nodes, hidden_dim)
    y = torch.zeros(num_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

def build_loader(edge_list, batch_size, hidden_dim):
    data = build_graph_from_interactions(edge_list, hidden_dim)
    dataset = [data]  # DataLoader cần list-like dataset
    return DataLoader(dataset, batch_size=batch_size)

# Normalize Layer
class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if norm == 'none' or dim is None:
            self.norm = lambda x: x
        elif norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

# GIN Layer
def make_gin_conv(input_dim: int, out_dim: int):
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return GINConv(mlp)

# GConv backbone
class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, activation, encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.layers = torch.nn.ModuleList([make_gin_conv(input_dim, hidden_dim)] +
                                          [make_gin_conv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.batch_norm = Normalize(hidden_dim, encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            self.activation = torch.nn.ReLU()
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)

# Encoder with momentum update
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super().__init__()
        self.online_encoder = encoder
        self.augmentor = augmentor
        self.target_encoder = None
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum=0.99):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            p.data = momentum * p.data + (1 - momentum) * new_p.data

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        aug1, aug2 = self.augmentor
        x1, ei1, ew1 = aug1(x, edge_index, edge_weight)
        x2, ei2, ew2 = aug2(x, edge_index, edge_weight)
        h1, h1_online = self.online_encoder(x1, ei1, ew1)
        h2, h2_online = self.online_encoder(x2, ei2, ew2)
        g1 = global_add_pool(h1, batch)
        g2 = global_add_pool(h2, batch)
        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)
        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, ei1, ew1)
            _, h2_target = self.get_target_encoder()(x2, ei2, ew2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)
        return g1, g2, h1_pred, h2_pred, g1_target, g2_target

# Train loop
def train(encoder_model, contrast_model, dataloader, optimizer, momentum):
    encoder_model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32).to(data.batch.device)
        optimizer.zero_grad()
        _, _, h1_pred, h2_pred, g1_target, g2_target = encoder_model(data.x, data.edge_index, batch=data.batch)
        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred,
                              g1_target=g1_target.detach(), g2_target=g2_target.detach(), batch=data.batch)
        loss.backward()
        optimizer.step()
        encoder_model.update_target_encoder()
        total_loss += loss.item()
    return total_loss

# Test encoder representation
def test(encoder_model, dataloader):
    encoder_model.eval()
    xs, ys = [], []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32).to(data.batch.device)
        g1, g2, *_ = encoder_model(data.x, data.edge_index, batch=data.batch)
        xs.append(torch.cat([g1, g2], dim=1))
        ys.append(data.y)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    split = get_split(num_samples=x.size(0), train_ratio=0.8, test_ratio=0.1)
    return SVMEvaluator(linear=True)(x, y, split)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data from configuration files...")
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'BGRL', 'type': 'graph'},
        'output': './results/',
        'item.ranking.topN': [10, 20, 30, 50]
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print("\nBGRL Hyperparameter Tuning Framework\n" + "="*80)
    interaction = Interaction(base_config, train_set, test_set)
    data = build_movielens_graph(interaction).to(device)

    grid = {
        'hidden_dim': [16, 32, 64, 128, 256, 512],
        'num_layers': [1, 2, 3, 4],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'lr': [1e-1, 1e-2, 1e-3, 1e-4],
        'batch_size': [16, 32, 64, 128, 256, 512],
        'edge_p': [0.1, 0.2, 0.3],
        'feat_p': [0.1, 0.2, 0.3],
        'momentum': [0.9, 0.99, 0.999],
        'weight_decay': [1e-4, 1e-5, 1e-6],
        'activation': [torch.nn.ReLU, torch.nn.PReLU, torch.nn.LeakyReLU ]
    }
    default = {
        'hidden_dim': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'lr': 1e-2,
        'batch_size': 128,
        'edge_p': 0.2,
        'feat_p': 0.1,
        'momentum': 0.99,
        'weight_decay': 1e-5,
        'activation': torch.nn.ReLU
    }

    total_runs = sum(len(v) for v in grid.values())
    print(f"\nTotal combinations: {total_runs}\n" + '='*80)
    results = []
    run_count = 0
    for key, values in grid.items():
        print(f"\n{'='*80}\n Tuning hyperparameter: {key}")
        for val in values:
            run_count += 1
            print(f"\n>>> [{run_count}/{total_runs}] {key} = {val}")
            param_config = default.copy()
            param_config[key] = val
            # conf = make_config(param_config)

            aug1 = Compose([EdgeRemoving(pe=param_config['edge_p']), FeatureMasking(pf=param_config['feat_p'])])
            aug2 = Compose([EdgeRemoving(pe=param_config['edge_p']), FeatureMasking(pf=param_config['feat_p'])])
            gconv = GConv(input_dim=param_config['hidden_dim'],
                          hidden_dim=param_config['hidden_dim'],
                          num_layers=param_config['num_layers'],
                          dropout=param_config['dropout'],
                          activation=param_config['activation']
                          ).to(device)
            encoder = Encoder(encoder=gconv,
                              augmentor=(aug1, aug2),
                              hidden_dim=param_config['hidden_dim']
                              ).to(device)
            contrast = BootstrapContrast(loss=BootstrapLatent(), mode='G2L').to(device)
            optimizer = Adam(encoder.parameters(), lr=param_config['lr'], weight_decay=param_config['weight_decay'])
            # dataloader = DataLoader(dataset, batch_size=batch_size)
            best_metrics = None
            for epoch in range(100):
                loss = train(encoder, contrast,
                            build_loader(train_set, param_config['batch_size'], param_config['hidden_dim']),
                            optimizer, momentum=param_config['momentum'])
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

                recommender = GraphRecommender(base_config, train_set, test_set)
                fast_result = recommender.fast_evaluation(epoch, topK=base_config.get('item.ranking.topN', [10, 20, 30, 50]))
                print(f"(Fast Eval) Recall={fast_result['Recall']:.4f}")

                metrics = recommender.evaluate(recommender.test())
                # print("\n".join(metrics))
                # metrics = GraphRecommender.evaluate(encoder, train_set, test_set, topK=base_config['item.ranking.topN'])
                # print(f"(Eval) " +
                # ", ".join([f"Recall@{k}={metrics['Recall'][k]:.4f}, "
                #            f"Precision@{k}={metrics['Precision'][k]:.4f}, "
                #            f"NDCG@{k}={metrics['NDCG'][k]:.4f}, "
                #            f"HR@{k}={metrics['HR'][k]:.4f}"
                #            for k in base_config['item.ranking.topN']]))
                best_metrics = metrics
            results.append({'param': param_config, 'metrics': best_metrics})

    with open('bgrl_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()