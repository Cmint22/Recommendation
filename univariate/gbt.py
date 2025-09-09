# !pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
# !pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu118

# !pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# !pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# !pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# !pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# !pip install torch-geometric
# !pip uninstall -y pytorch-lightning
# !pip install pytorch-lightning==1.5.0

import os, json, math, copy, torch, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import torch_geometric.transforms as T
from tqdm import tqdm
from torch.optim import Adam
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
from typing import Optional, Tuple, NamedTuple, List
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import CosineAnnealingLR



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

    def predict(self, user):
        num_items = len(self.data.item)
        scores = np.random.rand(num_items)
        return scores

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


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss


def bt_loss(h1: torch.Tensor, h2: torch.Tensor, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)
    if lambda_ is None:
        lambda_ = 1. / feature_dim
    if batch_norm:
        z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
        z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
    else:
        c = h1.T @ h2 / batch_size
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()
    return loss


class BarlowTwins(Loss):
    def __init__(self, lambda_: float = None, batch_norm: bool = True, eps: float = 1e-5):
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        loss = bt_loss(anchor, sample, self.lambda_, self.batch_norm, self.eps)
        return loss.mean()


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


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
 

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


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split
        result = self.evaluate(x, y, split)
        return result


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()
        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()
                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])
                loss.backward()
                optimizer.step()
                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')
                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch
                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)
        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro
        }


class WithinEmbedContrast(torch.nn.Module):
    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1, h2):
        l1 = self.loss(anchor=h1, sample=h2, **self.kwargs)
        l2 = self.loss(anchor=h2, sample=h1, **self.kwargs)
        return (l1 + l2) * 0.5



class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_features, activation, momentum):
        super(GConv, self).__init__()
        self.num_features = num_features
        # input_dim = input_dim * num_features
        expanded_input_dim = input_dim * num_features
        self.act = activation
        self.bn = torch.nn.BatchNorm1d(num_features * hidden_dim, momentum=momentum)
        self.conv1 = GCNConv(expanded_input_dim, num_features * hidden_dim, cached=False)
        self.conv2 = GCNConv(num_features * hidden_dim, hidden_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        if self.num_features > 1:
            x = x.repeat(1, self.num_features)
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2


def train(encoder_model, contrast_model, data, optimizer, momentum):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(z1, z2) + momentum
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data from configuration files...")
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'G-BT', 'type': 'graph'},
        'output': './results/',
        'item.ranking.topN': [10, 20, 30, 50]
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print("\nG-BT Hyperparameter Tuning Framework\n" + "="*80)

    grid = {
        'num_features': [1, 2, 3, 4],
        'activation': [torch.nn.ReLU(), torch.nn.PReLU(), torch.nn.ELU()],
        'momentum': [0.01, 0.05, 0.1],
        'pe': [0.1, 0.3, 0.5],
        'pf': [0.1, 0.2, 0.3],
        'hidden_dim': [128, 256, 512, 1024],
        'lr': [1e-4, 5e-4, 1e-3, 5e-3]
    }
    default = {
        'num_features': 2,
        'activation': torch.nn.PReLU(),
        'momentum': 0.01,
        'pe': 0.5,
        'pf': 0.1,
        'hidden_dim': 256,
        'lr': 5e-4
    }

    total_runs = sum(len(v) for v in grid.values())
    print(f"\nTotal combinations : {total_runs}\n" + '='*80)
    results = []
    run_count = 0
    path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
    dataset = WikiCS(path, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    for key, values in grid.items():
        print(f"\n{'='*80}\n Tuning hyperparameter: {key}")
        for val in values:
            run_count += 1
            print(f"\n>>> [{run_count}/{total_runs}] {key} = {val}")
            param_config = default.copy()
            param_config[key] = val
            aug1 = Compose([EdgeRemoving(pe=param_config['pe']), FeatureMasking(pf=param_config['pf'])])
            aug2 = Compose([EdgeRemoving(pe=param_config['pe']), FeatureMasking(pf=param_config['pf'])])
            gconv = GConv(input_dim=data.num_features, 
                          hidden_dim=param_config['hidden_dim'],
                          num_features=param_config['num_features'],
                          activation=param_config['activation'],
                          momentum=param_config['momentum']).to(device)
            encoder = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
            contrast = WithinEmbedContrast(loss=BarlowTwins()).to(device)
            optimizer = Adam(encoder.parameters(), lr=param_config['lr'])
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=4000)
            best_metrics = None
            for epoch in range(1):
                loss = train(encoder, contrast, data, optimizer, momentum=param_config['momentum'])
                scheduler.step()
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

            recommender = GraphRecommender(base_config, train_set, test_set)
            fast_result = recommender.fast_evaluation(epoch, topK=base_config.get('item.ranking.topN', [10, 20, 30, 50]))
            print(f"(Fast Eval): Recall={fast_result['Recall']:.4f}")

            metrics = recommender.evaluate(recommender.test())
            best_metrics = metrics
        results.append({'param': param_config, 'metrics': best_metrics})
            
    with open('g-bt_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)



if __name__ == '__main__':
    main()