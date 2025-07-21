import os, json, math, copy
from random import shuffle, choice
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TorchGraphInterface:
    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape).to(device)


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
            self.bestPerformance = [ epoch+1, performance]
        return performance



class BUIR(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BUIR, self).__init__(conf, training_set, test_set)
        args = self.config['BUIR']
        self.momentum = float(args['tau'])
        self.n_layers = int(args['n_layer'])
        self.drop_rate = float(args['drop_rate'])
        self.emb_size = conf.get("emb_size", 64)
        self.lRate = conf.get("lr", 0.001)
        self.batch_size = conf.get("batch_size", 2048)
        self.model = BUIR_NB(self.data, self.emb_size, self.momentum, self.n_layers, self.drop_rate, True)

    def train(self):
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(1):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, i_idx, j_idx = batch
                inputs = {'user': user_idx, 'item': i_idx}
                model.train()
                output = model(inputs)
                batch_loss = model.get_loss(output)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                model.update_target(user_idx,i_idx)
                if (n + 1) % 100 == 0:
                    print('Training:', epoch + 1, 'Batch', n+1, 'Batch_loss:', batch_loss.item())
            model.eval()
            self.p_u_online, self.u_online, self.p_i_online, self.i_online = self.model.get_embedding()
            self.fast_evaluation(epoch)
            self.save()
        self.p_u_online, self.u_online, self.p_i_online, self.i_online = self.best_p_u, self.best_u, self.best_p_i, self.best_i
        # if hasattr(self, 'best_p_u'):
        #     self.p_u_online, self.u_online, self.p_i_online, self.i_online = \
        #         self.best_p_u, self.best_u, self.best_p_i, self.best_i
        # return self.bestPerformance[1] if self.bestPerformance else {}
        return self.evaluate()

    def save(self):
        self.best_p_u, self.best_u, self.best_p_i, self.best_i = self.model.get_embedding()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score_ui = torch.matmul(self.p_u_online[u], self.i_online.transpose(0, 1))
        score_iu = torch.matmul(self.u_online[u], self.p_i_online.transpose(0, 1))
        score = score_ui + score_iu
        return score.cpu().numpy()

    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print("Detailed TopN Evaluation:")
        print("".join(metrics))
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':', 1)]}



class BUIR_NB(nn.Module):
    def __init__(self, data, emb_size, momentum, n_layers, drop_rate, drop_flag=False):
        super(BUIR_NB, self).__init__()
        self.emb_size = emb_size
        self.momentum = momentum
        self.online_encoder = LGCN_Encoder(data, emb_size, n_layers, drop_rate, drop_flag)
        self.target_encoder = LGCN_Encoder(data, emb_size, n_layers, drop_rate, drop_flag)
        self.predictor = nn.Linear(emb_size, emb_size)
        self._init_target()

    def _init_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def update_target(self,u_idx,i_idx):
        # for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
        #     param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)
        self.target_encoder.embedding_dict['user_emb'].data[u_idx] = self.target_encoder.embedding_dict['user_emb'].data[u_idx]*self.momentum\
                                                              + self.online_encoder.embedding_dict['user_emb'].data[u_idx]*(1-self.momentum)
        self.target_encoder.embedding_dict['item_emb'].data[i_idx] = self.target_encoder.embedding_dict['item_emb'].data[i_idx]*self.momentum\
                                                              + self.online_encoder.embedding_dict['item_emb'].data[i_idx]*(1-self.momentum)

    def forward(self, inputs):
        u_online, i_online = self.online_encoder(inputs)
        u_target, i_target = self.target_encoder(inputs)
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output
        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)
        return (loss_ui + loss_iu).mean()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, drop_rate, drop_flag=False):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.drop_ratio = drop_rate
        self.drop_flag = drop_flag
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(device)
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        A_hat = self.sparse_dropout(self.sparse_norm_adj, np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        users, items = inputs['user'], inputs['item']
        user_embeddings = user_all_embeddings[users]
        item_embeddings = item_all_embeddings[items]
        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
    

class BUIRTuner:
    def __init__(self, train_set, test_set, base_config):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_config
        self.results = []
        self. grid = { 
            'factors': [16, 32, 64, 128, 256, 512],
            'batch_size': [128, 256, 512, 1024, 2048, 4096], 
            'emb_size': [16, 32, 64, 128, 256, 512], 
            'lambda': [0.00001, 0.0001, 0.001, 0.01, 0.1], 
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'BUIR.n_layer': [1, 2, 3, 4], 
            'BUIR.tau': [0.1, 0.5, 1.0], 
            'BUIR.drop_rate': [0.1, 0.2, 0.3] 
        }
        self.default = {
            'factors': 64,
            'batch_size': 2048,
            'emb_size': 64,
            'lambda': 0.0001,
            'lr': 0.001,
            'BUIR.n_layer': 2,
            'BUIR.tau': 1.0,
            'BUIR.drop_rate': 0.2
        }

    def run(self):
        total_runs = sum(len(v) for v in self.grid.values())
        print(f"\nTotal combinations: {total_runs}\n" + '='*80)
        run_count = 0
        for key, values in self.grid.items():
            print(f"\n{'='*80}\n Tuning hyperparameter: {key}")
            for val in values:
                run_count += 1
                print(f"\n>>> [{run_count}/{total_runs}] {key} = {val}")
                param_config = self.default.copy()
                param_config[key] = val
                conf = self.make_config(param_config)
                try:
                    model = BUIR(conf, self.train_set, self.test_set)
                    metrics = model.train()
                    print(f"Result: NDCG: {metrics.get('NDCG')}, Recall: {metrics.get('Recall')}, "
                          f"Precision: {metrics.get('Precision')}, Hit Ratio: {metrics.get('Hit Ratio')}")
                    self.results.append({'config': conf, 'metrics': metrics})
                except Exception as e:
                    import traceback
                    print(f"Error tuning {key} = {val}: {e}")
                    traceback.print_exc()
        with open('buir_tuning_individual.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'factors': params['factors'],
            'batch_size': params['batch_size'],
            'emb_size': params['emb_size'],
            'lambda': params['lambda'],
            'lr': params['lr'],
            'max.epoch': 1,
            'item.ranking.topN': [10, 20, 30, 50],
            'BUIR':{
                'n_layer': params['BUIR.n_layer'],
                'tau': params['BUIR.tau'],
                'drop_rate': params['BUIR.drop_rate']
            }
        })
        for key in ['n_layer', 'tau', 'drop_rate']:
            conf[f'BUIR.{key}'] = conf['BUIR'][key]
        return conf


def load_data(path):
    return [[*line.strip().split()[:2], 1.0] for line in open(path) if line.strip()] if os.path.exists(path) else []



if __name__ == '__main__':
    print("Loading data from configuration files...")
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'model': {'name': 'BUIR', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print("\nBUIR Hyperparameter Tuning Framework\n" + "="*80)
    tuner = BUIRTuner(train_set, test_set, base_config)
    tuner.run()