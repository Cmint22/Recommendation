import os, sys, torch, json
import numpy as np
from torch import nn
from datetime import datetime
import itertools, copy
from util.loss_torch import l2_reg_loss, InfoNCE, batch_softmax_loss
from util.evaluation import ranking_evaluation
from util.sampler import next_batch_pairwise
from base.graph_recommender import GraphRecommender

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SSL4RecModel(GraphRecommender):
    def __init__(self, conf, train_data, test_data):
        super().__init__(conf, train_data, test_data)
        args = conf['SSL4Rec']
        self.cl_rate = args['alpha']
        self.tau = args['tau']
        self.drop = args['drop']
        self.model = DNNEncoder(self.data, self.emb_size, self.drop, self.tau)
        self.best_performance = {}

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate)
        best_epoch, best_metric, no_improv = 0, {}, 0
        for epoch in range(self.maxEpoch):
            self.model.train()
            losses = []
            for batch in next_batch_pairwise(self.data, self.batch_size):
                u, i, _ = batch
                u_emb, i_emb = self.model(u, i)
                rec_loss = batch_softmax_loss(u_emb, i_emb, self.tau)
                cl_loss = self.cl_rate * self.model.cal_cl_loss(i)
                loss = rec_loss + cl_loss + l2_reg_loss(self.reg, u_emb, i_emb)

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                self.query_emb, self.item_emb = self.model(
                    range(self.data.user_num), range(self.data.item_num))
            current = self.evaluate()
            if not best_metric or self.is_better(current, best_metric):
                best_metric, best_epoch = current, epoch
                self.save()
                no_improv = 0
            else:
                no_improv += 1
            if no_improv >= 3: break
        return best_metric

    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        result = {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':')]}
        return result

    def is_better(self, current, best): return current.get('Recall', 0) > best.get('Recall', 0)

    def save(self):
        with torch.no_grad():
            self.best_query_emb, self.best_item_emb = self.model(
                list(range(self.data.user_num)), list(range(self.data.item_num)))

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(self.query_emb[u], self.item_emb.T).cpu().numpy()


class DNNEncoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, tau):
        super().__init__()
        self.emb_size = emb_size
        self.tau = tau
        self.dropout = nn.Dropout(drop_rate)
        init = nn.init.xavier_uniform_
        self.initial_user = nn.Parameter(init(torch.empty(data.user_num, emb_size)))
        self.initial_item = nn.Parameter(init(torch.empty(data.item_num, emb_size)))
        self.user_net = nn.Sequential(nn.Linear(emb_size, 1024), nn.ReLU(), nn.Linear(1024, 128), nn.Tanh())
        self.item_net = nn.Sequential(nn.Linear(emb_size, 1024), nn.ReLU(), nn.Linear(1024, 128), nn.Tanh())

    def forward(self, u, i):
        return self.user_net(self.initial_user[u]), self.item_net(self.initial_item[i])

    def cal_cl_loss(self, i):
        emb = self.initial_item[i]
        i1, i2 = self.dropout(emb), self.dropout(emb)
        return InfoNCE(self.item_net(i1), self.item_net(i2), self.tau)


class Tuner:
    def __init__(self, train_set, test_set, base_conf):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_conf
        self.results = []
        self.grid = {
            'embedding.size': [32, 64, 128],
            'batch.size': [1024, 2048, 4096],
            'learning.rate': [0.001],
            'reg.lambda': [0.0001],
            'SSL4Rec.tau': [0.07, 0.1, 0.2],
            'SSL4Rec.alpha': [0.1, 0.2, 0.3],
            'SSL4Rec.drop': [0.1, 0.2, 0.3]
        }

    def run(self):
        keys, vals = list(self.grid.keys()), list(self.grid.values())
        for i, combo in enumerate(itertools.product(*vals), 1):
            conf = self.make_config(dict(zip(keys, combo)))
            try:
                print(f"[{i}] Training: {conf['SSL4Rec']}")
                model = SSL4RecModel(conf, self.train_set, self.test_set)
                metrics = model.train()
                self.results.append({'config': conf, 'metrics': metrics})
            except Exception as e:
                print(f"Error in combo {i}: {e}")
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
            'SSL4Rec': {
                'tau': params['SSL4Rec.tau'],
                'alpha': params['SSL4Rec.alpha'],
                'drop': params['SSL4Rec.drop']
            }
        })
        return conf

    def best(self, metric='Recall'):
        valid = [r for r in self.results if 'metrics' in r]
        return max(valid, key=lambda r: r['metrics'].get(metric, 0), default=None)

    def save(self, path='ssl4rec_results.json'):
        with open(path, 'w') as f: json.dump(self.results, f, indent=2)
        print(f"Saved results to {path}")


# === Utility: Load data ===
def load_data(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [[*line.strip().split()[:2], 1.0] for line in f if line.strip()]


# === Main Run ===
if __name__ == '__main__':
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'SSL4Rec', 'type': 'graph'},
        'output': './results/ml100k'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    tuner = Tuner(train_set, test_set, base_config)
    tuner.run()
    best = tuner.best()
    if best:
        print(f"\nBest config:\n{json.dumps(best['config'], indent=2)}")
        print(f"Best metrics: {best['metrics']}")
    tuner.save()