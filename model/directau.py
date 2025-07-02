import os
import json
import copy
import torch
import itertools
import torch.nn.functional as F
from datetime import datetime
from torch import nn
from util.sampler import next_batch_pairwise
from util.loss_torch import l2_reg_loss
from util.evaluation import ranking_evaluation
from base.graph_recommender import GraphRecommender
from base.torch_interface import TorchGraphInterface


class DirectAU(GraphRecommender):
    def __init__(self, conf, train_set, test_set):
        super().__init__(conf, train_set, test_set)
        args = self.config['DirectAU']
        self.gamma = float(args['gamma'])
        self.n_layers = int(args['n_layers'])
        self.reg = conf.get("reg.lambda", 0.0001)
        self.batch_size = conf.get("batch.size", 512)
        self.emb_size = conf.get("embedding.size", 64)
        self.topN = conf.get("item.ranking.topN", [10, 20, 30, 50])
        self.lRate = conf.get("learning.rate", 0.001)
        self.model = LGCNEncoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(1):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                user_emb, item_emb, _ = model()
                u_emb = user_emb[user_idx]
                p_emb = item_emb[pos_idx]
                loss = self.calculate_loss(u_emb, p_emb)
                loss += l2_reg_loss(self.reg, u_emb, p_emb) / self.batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (n + 1) % 100 == 0:
                    align = self.alignment(u_emb, p_emb).item()
                    uniform = self.gamma * (self.uniformity(u_emb).item() + self.uniformity(p_emb).item()) / 2
                    print(f"Batch {n+1}, align: {align:.4f}, uniform: {uniform:.4f}, total loss: {loss.item():.4f}")
        with torch.no_grad():
            self.user_emb, self.item_emb, _ = model()
            self.user_emb = self.user_emb.detach()
            self.item_emb = self.item_emb.detach()
        return self.evaluate()

    def calculate_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = self.gamma * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align + uniform

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        pdist = torch.pdist(x, p=2)
        if pdist.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        return (pdist.pow(2).mul(-t).exp().mean() + 1e-8).log()

    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':', 1)]}

    def predict(self, u):
        u = self.data.get_user_id(u)
        if not hasattr(self, 'user_emb') or not hasattr(self, 'item_emb'):
            self.user_emb, self.item_emb, _ = self.model()
            self.user_emb = self.user_emb.detach()
            self.item_emb = self.item_emb.detach()
        return torch.matmul(self.user_emb[u], self.item_emb.T).detach().cpu().numpy()

    def is_better(self, current, best):
        return current.get('Recall', 0) > best.get('Recall', 0)


class LGCNEncoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super().__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cpu()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        return nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
        })

    def forward(self):
        emb = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_emb = [emb]
        for _ in range(self.layers):
            emb = torch.sparse.mm(self.sparse_norm_adj, emb)
            all_emb.append(emb)
        final_emb = torch.stack(all_emb).mean(0)
        return final_emb[:self.data.user_num], final_emb[self.data.user_num:], all_emb


class DirectAUTuner:
    def __init__(self, train_set, test_set, base_config):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_config
        self.results = []
        self.grid = {
            'embedding.size': [64, 128],
            'batch.size': [512, 1024, 2048],
            'learning.rate': [0.001],
            'reg.lambda': [0.0001],
            'DirectAU.gamma': [0.005, 0.01, 0.05, 0.1],
            'DirectAU.n_layers': [2, 3]
        }

    def run(self):
        keys, vals = list(self.grid.keys()), list(self.grid.values())
        best_result = None
        for i, combo in enumerate(itertools.product(*vals), 1):
            conf = self.make_config(dict(zip(keys, combo)))
            try:
                print(f"[{i}] Training: embedding={conf['embedding.size']}, batch={conf['batch.size']}, lr={conf['learning.rate']}, reg={conf['reg.lambda']}, gamma={conf['DirectAU']['gamma']}, n_layers={conf['DirectAU']['n_layers']}")
                model = DirectAU(conf, self.train_set, self.test_set)
                metrics = model.train()
                print(f"Metrics for config {i}:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.5f}")
                print("-" * 60)
                self.results.append({'config': conf, 'metrics': metrics})
                result = {'config': conf, 'metrics': metrics}
                self.results.append(result)
                if best_result is None or metrics.get('Recall', 0) > best_result['metrics'].get('Recall', 0):
                    best_result = result
            except Exception as e:
                print(f"Error in config {i}: {e}")
                self.results.append({'config': conf, 'error': str(e)})

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'embedding.size': params['embedding.size'],
            'batch.size': params['batch.size'],
            'learning.rate': params['learning.rate'],
            'reg.lambda': params['reg.lambda'],
            'max.epoch': 1,
            'item.ranking.topN': [10, 20, 30, 50],
            'DirectAU': {
                'gamma': params['DirectAU.gamma'],
                'n_layers': params['DirectAU.n_layers']
            }
        })
        return conf


def load_data(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [[*line.strip().split()[:2], 1.0] for line in f if line.strip()]


if __name__ == '__main__':
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'DirectAU', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    tuner = DirectAUTuner(train_set, test_set, base_config)
    tuner.run()
