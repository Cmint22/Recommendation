import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import torch
import faiss
import copy
import itertools
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.evaluation import ranking_evaluation
from base.graph_recommender import GraphRecommender
from base.torch_interface import TorchGraphInterface


class NCLModel(GraphRecommender):
    def __init__(self, conf, train_set, test_set):
        super(NCLModel, self).__init__(conf, train_set, test_set)
        args = self.config['NCL']
        self.n_layers = args['n_layer']
        self.ssl_temp = args['tau']
        self.ssl_reg = args['ssl_reg']
        self.proto_reg = args['proto_reg']
        self.hyper_layers = args['hyper_layers']
        self.alpha = args['alpha']
        self.k = args['num_clusters']
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
            # print(f"{'='*60}\nTraining Epoch 1\n{'='*60}")
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb, rec_item_emb, emb_list = model()
                user_emb = rec_user_emb[user_idx]
                pos_emb = rec_item_emb[pos_idx]
                neg_emb = rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)
                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers * 2]
                ssl_loss = self.ssl_layer_loss(context_emb, initial_emb, user_idx, pos_idx)
                # self.e_step()
                proto_loss = self.ProtoNCE_loss(initial_emb, user_idx, pos_idx)
                total_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_emb, neg_emb) / self.batch_size + ssl_loss + proto_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if (n + 1) % 100 == 0:
                    print(f"Batch {n+1}: rec_loss={rec_loss.item():.4f}, ssl_loss={ssl_loss.item():.4f}, proto_loss={proto_loss.item():.4f}, total_loss={total_loss.item():.4f}")
        self.model.eval()
        with torch.no_grad():
            self.user_emb, self.item_emb, _ = self.model()
        return self.evaluate()

    def e_step(self):
        user_emb, item_emb, _ = self.model()
        user_np = user_emb.detach().cpu().numpy()
        item_np = item_emb.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_np)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_np)

    def run_kmeans(self, x):
        if x.shape[0] < self.k * 39:
            self.k = min(self.k, max(10, x.shape[0] // 10))
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=False)
        kmeans.train(x)
        centroids = torch.tensor(kmeans.centroids)
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
        user2centroids = self.user_centroids[self.user_2cluster[user_idx]]
        item2centroids = self.item_centroids[self.item_2cluster[item_idx]]
        loss_user = InfoNCE(user_emb[user_idx], user2centroids, self.ssl_temp) * self.batch_size
        loss_item = InfoNCE(item_emb[item_idx], item2centroids, self.ssl_temp) * self.batch_size
        return self.proto_reg * (loss_user + loss_item)

    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        # print("Evaluation metrics:", metrics)
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
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cpu()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        emb = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
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
        self.grid = {
            'embedding.size': [64, 128],
            'batch.size': [256, 512, 1024, 2048],
            'learning.rate': [0.001],
            'reg.lambda': [0.0001],
            'NCL.tau': [0.05, 0.1],
            'NCL.alpha': [0.5, 1.0],
            'NCL.num_clusters': [500, 1000]
        }

    def run(self):
        keys, vals = list(self.grid.keys()), list(self.grid.values())
        best_result = None
        for i, combo in enumerate(itertools.product(*vals), 1):
            conf = self.make_config(dict(zip(keys, combo)))
            try:
                print(f"[{i}] Training: {conf['NCL']}")
                model = NCLModel(conf, self.train_set, self.test_set)
                metrics = model.train()
                print(f"Metrics fo config {i}:")
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

        # Ghi duy nhất kết quả tốt nhất
        if best_result:
            with open('ncl_results.json', 'w') as f:
                json.dump([best_result], f, indent=2)
            print("Saved best result to ncl_results.json")

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'embedding.size': params['embedding.size'],
            'batch.size': params['batch.size'],
            'learning.rate': params['learning.rate'],
            'reg.lambda': params['reg.lambda'],
            'max.epoch': 120,
            'item.ranking.topN': [10, 20, 30, 50],
            'NCL': {
                'tau': params['NCL.tau'],
                'alpha': params['NCL.alpha'],
                'num_clusters': params['NCL.num_clusters'],
                'n_layer': 3,
                'ssl_reg': 1e-5,
                'proto_reg': 1e-5,
                'hyper_layers': 1
            }
        })
        return conf

    def best(self, metric='Recall'):
        valid = [r for r in self.results if 'metrics' in r]
        return max(valid, key=lambda r: r['metrics'].get(metric, 0), default=None)

    def save(self, path='ncl_results.json'):
        best = self.best()
        if best:
            with open(path, 'w') as f:
                json.dump(best, f, indent=2)
            print(f"Saved best result to {path}")
        else:
            print("No valid result to save.")


def load_data(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [[*line.strip().split()[:2], 1.0] for line in f if line.strip()]


if __name__ == '__main__':
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'NCL', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    tuner = Tuner(train_set, test_set, base_config)
    tuner.run()
    # best = tuner.best()
    # if best:
    #     print(f"\nBest config:\n{json.dumps(best['config'], indent=2)}")
    #     print(f"Best metrics: {best['metrics']}")
    # tuner.save()