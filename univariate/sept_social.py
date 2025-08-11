import tensorflow as tf
from scipy.sparse import eye
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier
import os, random, torch, math, copy, json
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
class TFGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(device)

    @staticmethod
    def convert_sparse_mat_to_tensor_inputs(X):
        coo = X.tocoo()
        indices = np.vstack((coo.row, coo.col)).T
        return indices, coo.data, coo.shape


# ---------------------- Graph Augmentor ------------------------------
class GraphAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def edge_dropout(sp_adj, drop_rate):
        row, col = sp_adj.nonzero()
        idx = np.arange(len(row))
        keep_idx = np.random.choice(idx, int(len(idx) * (1 - drop_rate)), replace=False)
        new_row, new_col = row[keep_idx], col[keep_idx]
        data = np.ones_like(new_row, dtype=np.float32)
        return sp.csr_matrix((data, (new_row, new_col)), shape=sp_adj.shape)


# ------------------------------ Graph ---------------------------------
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

    def convert_to_laplacian_mat(self, adj_mat):
        pass


# ---------------------------- Relation ------------------------------
class Relation(Graph):
    def __init__(self, conf, relation, user):
        super().__init__()
        self.config = conf
        self.social_user = {}
        self.relation = relation
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self.user = user
        self.__initialize()

    def __initialize(self):
        idx = []
        for n, pair in enumerate(self.relation):
            if pair[0] not in self.user or pair[1] not in self.user:
                idx.append(n)
        for item in reversed(idx):
            del self.relation[item]
        for line in self.relation:
            user1, user2, weight = line
            # add relations to dict
            self.followees[user1][user2] = weight
            self.followers[user2][user1] = weight

    def get_social_mat(self):
        row, col, entries = [], [], []
        for pair in self.relation:
            row += [self.user[pair[0]]]
            col += [self.user[pair[1]]]
            entries += [1.0]
        social_mat = sp.csr_matrix((entries, (row, col)), shape=(len(self.user), len(self.user)), dtype=np.float32)
        return social_mat

    def get_birectional_social_mat(self):
        social_mat = self.get_social_mat()
        bi_social_mat = social_mat.multiply(social_mat)
        return bi_social_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        (row_np_keep, col_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (row_np_keep, col_np_keep)), shape=adj_shape, dtype=np.float32)
        return self.normalize_graph_mat(tmp_adj)

    def weight(self, u1, u2):
        if u1 in self.followees and u2 in self.followees[u1]:
            return self.followees[u1][u2]
        else:
            return 0

    def get_followers(self, u):
        if u in self.followers:
            return self.followers[u]
        else:
            return {}

    def get_followees(self, u):
        if u in self.followees:
            return self.followees[u]
        else:
            return {}

    def has_followee(self, u1, u2):
        if u1 in self.followees:
            if u2 in self.followees[u1]:
                return True
            else:
                return False
        return False

    def has_follower(self, u1, u2):
        if u1 in self.followers:
            if u2 in self.followers[u1]:
                return True
            else:
                return False
        return False

    def size(self):
        return len(self.followers), len(self.relation)
    

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
        self.interaction_mat = self.__create_sparse_interaction_matrix()

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

    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, user):
        return self.user[user]

    def user_rated(self, user):
        return list(self.training_set_u[user]), []


# ---------------------------- Base Recommender ----------------------------
class Recommender(nn.Module):
    def __init__(self, conf, train, test):
        super().__init__()
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


# ------------------------------- SEPT Model  -------------------------------
class SEPT(GraphRecommender):
    def __init__(self, conf, training_set, test_set, social_data, **kwargs):
        super().__init__(conf, training_set, test_set, **kwargs)
        args = self.config['SEPT']
        self.n_layers = int(args['n_layer'])
        self.ss_rate = float(args['ss_rate'])
        self.drop_rate = float(args['drop_rate'])
        self.instance_cnt = int(args['ins_cnt'])
        self.emb_size = conf.get('emb_size', 64)
        self.batch_size = conf.get('batch_size', 2048)
        self.lRate = conf.get('lr', 0.001)
        self.reg = conf.get('reg_lambda', 0.0001)
        self.maxEpoch = int(self.config['max.epoch'])
        self.social_data = Relation(conf, social_data, self.data.user)

        self.user_embeddings = nn.Parameter(torch.empty(self.data.user_num, self.emb_size))
        self.item_embeddings = nn.Parameter(torch.empty(self.data.item_num, self.emb_size))
        xavier(self.user_embeddings)
        xavier(self.item_embeddings)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lRate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def print_model_info(self):
        super().print_model_info()
        print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
        print('=' * 80)

    def get_social_related_views(self, social_mat, interaction_mat):
        social_matrix = social_mat.dot(social_mat)
        social_matrix = social_matrix.multiply(social_mat) + eye(self.data.user_num, dtype=np.float32)
        sharing_matrix = interaction_mat.dot(interaction_mat.T)
        sharing_matrix = sharing_matrix.multiply(social_mat) + eye(self.data.user_num, dtype=np.float32)
        social_matrix = self.social_data.normalize_graph_mat(social_matrix)
        sharing_matrix = self.social_data.normalize_graph_mat(sharing_matrix)
        return [social_matrix, sharing_matrix]

    def encoder(self, emb, adj, n_layers):
        all_embs = [emb]
        for _ in range(n_layers):
            emb = torch.sparse.mm(adj, emb)
            emb = F.normalize(emb)
            all_embs.append(emb)
        all_embs = torch.stack(all_embs, dim=0).sum(dim=0)
        return torch.split(all_embs, [self.data.user_num, self.data.item_num], 0)

    def social_encoder(self, emb, adj, n_layers):
        all_embs = [emb]
        for _ in range(n_layers):
            emb = torch.sparse.mm(adj, emb)
            emb = F.normalize(emb)
            all_embs.append(emb)
        return torch.stack(all_embs, dim=0).sum(dim=0)

    def build(self):
        self.bi_social_mat = self.social_data.get_birectional_social_mat()
        social_mat, sharing_mat = self.get_social_related_views(self.bi_social_mat, self.data.interaction_mat)
        self.social_mat = TFGraphInterface.convert_sparse_mat_to_tensor(social_mat).to(self.device)
        self.sharing_mat = TFGraphInterface.convert_sparse_mat_to_tensor(sharing_mat).to(self.device)
        self.norm_adj = TFGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).to(self.device)

    def label_prediction(self, emb, u_idx):
        unique_u = torch.unique(u_idx)
        emb = F.normalize(emb[unique_u])
        aug_emb = F.normalize(self.aug_user_embeddings[unique_u])
        prob = torch.matmul(emb, aug_emb.T)
        return F.softmax(prob, dim=1)

    def sampling(self, logits):
        return torch.topk(logits, self.instance_cnt, dim=1).indices

    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1 + prob2) / 2
        return self.sampling(positive)

    def neighbor_discrimination(self, positive, emb, u_idx):
        unique_u = torch.unique(u_idx)
        emb = F.normalize(emb[unique_u])
        aug_emb = F.normalize(self.aug_user_embeddings[unique_u])
        pos_emb = aug_emb[positive]

        emb2 = emb.unsqueeze(1).expand(-1, self.instance_cnt, -1)
        pos = torch.sum(emb2 * pos_emb, dim=2)
        ttl_score = torch.matmul(emb, aug_emb.T)
        pos_score = torch.sum(torch.exp(pos / 0.1), dim=1)
        ttl_score = torch.sum(torch.exp(ttl_score / 0.1), dim=1)
        ssl_loss = -torch.sum(torch.log(pos_score / ttl_score))
        return ssl_loss

    def train(self):
        self.build()
        for epoch in range(self.maxEpoch):
            if epoch > self.maxEpoch // 3:
                dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
                aug_mat = TFGraphInterface.convert_sparse_mat_to_tensor(self.data.convert_to_laplacian_mat(dropped_mat)).to(self.device)
            else:
                aug_mat = self.norm_adj

            for batch in next_batch_pairwise(self.data, self.batch_size):
                user_idx, pos_idx, neg_idx = map(lambda x: torch.tensor(x, dtype=torch.long, device=self.device), batch)
                ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], dim=0)
                self.rec_user_embeddings, self.rec_item_embeddings = self.encoder(ego_embeddings, self.norm_adj, self.n_layers)
                self.aug_user_embeddings, self.aug_item_embeddings = self.encoder(ego_embeddings, aug_mat, self.n_layers)
                self.sharing_view_embeddings = self.social_encoder(self.user_embeddings, self.sharing_mat, self.n_layers)
                self.friend_view_embeddings = self.social_encoder(self.user_embeddings, self.social_mat, self.n_layers)

                batch_user_emb = self.rec_user_embeddings[user_idx]
                batch_pos_item_emb = self.rec_item_embeddings[pos_idx]
                batch_neg_item_emb = self.rec_item_embeddings[neg_idx]

                rec_loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
                rec_loss += self.reg * (self.user_embeddings.norm(2).pow(2) + self.item_embeddings.norm(2).pow(2))

                if epoch > self.maxEpoch // 3:
                    social_prediction = self.label_prediction(self.friend_view_embeddings, user_idx)
                    sharing_prediction = self.label_prediction(self.sharing_view_embeddings, user_idx)
                    rec_prediction = self.label_prediction(self.rec_user_embeddings, user_idx)

                    f_pos = self.generate_pesudo_labels(sharing_prediction, rec_prediction)
                    sh_pos = self.generate_pesudo_labels(social_prediction, rec_prediction)
                    r_pos = self.generate_pesudo_labels(social_prediction, sharing_prediction)

                    neighbor_dis_loss = self.neighbor_discrimination(f_pos, self.friend_view_embeddings, user_idx)
                    neighbor_dis_loss += self.neighbor_discrimination(sh_pos, self.sharing_view_embeddings, user_idx)
                    neighbor_dis_loss += self.neighbor_discrimination(r_pos, self.rec_user_embeddings, user_idx)

                    total_loss = rec_loss + self.ss_rate * neighbor_dis_loss
                else:
                    total_loss = rec_loss

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

            with torch.no_grad():
                self.U = self.rec_user_embeddings.detach().cpu().numpy()
                self.V = self.rec_item_embeddings.detach().cpu().numpy()
                self.fast_evaluation(epoch)
        self.save()
        self.U, self.V = self.best_user_emb, self.best_item_emb
        return self.evaluate()

    def save(self):
        self.best_user_emb = self.rec_user_embeddings.detach().cpu().numpy()
        self.best_item_emb = self.rec_item_embeddings.detach().cpu().numpy()

    def predict(self, u):
        u = self.data.get_user_id(u)
        return np.dot(self.V, self.U[u])

    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print("Detailed TopN Evaluation:")
        print("".join(metrics))
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':', 1)]}
  

# ---------------------------- Hyperparameter Tuner ----------------------------
class SEPTTuner:
    def __init__(self, train_set, test_set, social_data, base_config):
        self.train_set, self.test_set, self.social_data = train_set, test_set, social_data
        self.base = base_config
        self.results = []
        self.grid = {
            'emb_size': [16, 32, 64, 128, 256, 512],
            'batch_size': [128, 256, 512, 1024, 2048, 4096],
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'reg_lambda': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'SEPT.n_layer': [1, 2, 3, 4],
            'SEPT.ss_rate': [0.001, 0.005, 0.01, 0.05],
            'SEPT.drop_rate': [0.1, 0.2, 0.3],
            'SEPT.ins_cnt': [5, 10, 15, 20]
        }
        self.default = {
            'emb_size': 64,
            'batch_size': 2048,
            'lr': 0.001,
            'reg_lambda':0.0001,
            'SEPT.n_layer': 2,
            'SEPT.ss_rate': 0.005,
            'SEPT.drop_rate': 0.3,
            'SEPT.ins_cnt': 10
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
                param = self.default.copy()
                param[key] = val
                conf = self.make_config(param)
                try:
                    model = SEPT(conf, self.train_set, self.test_set, self.social_data)
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
            'max.epoch': 1,
            'item.ranking.topN': [10, 20, 30, 50],
            'SEPT': {
                'n_layer': params['SEPT.n_layer'],
                'ss_rate': params['SEPT.ss_rate'],
                'drop_rate': params['SEPT.drop_rate'],
                'ins_cnt': params['SEPT.ins_cnt']
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
        'social.set': './dataset/ml100k/social.txt',
        'model': {'name': 'SEPT', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    social_data = load_data(base_config['social.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print(f"Loaded {len(social_data)} social interactions")
    print("\nSEPT Hyperparameter Tuning Framework\n" + "="*80)
    tuner = SEPTTuner(train_set, test_set, social_data, base_config)
    tuner.run()