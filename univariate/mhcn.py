from base.graph_recommender import GraphRecommender
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
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
    loss = -torch.log(10e-6 + torch.sigmoid(pos_scores - neg_scores))
    return torch.mean(loss)


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
            d_mat_inv =sp.diags(d_inv)
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
            # add relation to dict
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


# ------------------------------- MHCN Model  -------------------------------
class MHCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, social_data, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, **kwargs)
        args = self.config['MHCN']
        self.n_layers = int(args['n_layer'])
        self.ss_rate = float(args['ss_rate'])
        self.emb_size = conf.get('emb_size', 64)
        self.batch_size = conf.get('batch_size', 2048)
        self.lRate = conf.get('lr', 0.001)
        self.reg = conf.get('reg_lambda', 0.0001)
        self.maxEpoch = int(self.config['max.epoch'])
        self.social_data = Relation(conf, social_data, self.data.user)

    def print_model_info(self):
        super(MHCN, self).print_model_info()
        # # print social relation statistics
        print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
        print('=' * 80)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor using TFGraphInterface."""
        return TFGraphInterface.convert_sparse_mat_to_tensor(sparse_mx)
    
    def build_hyper_adj_mats(self):
        S = self.social_data.get_social_mat()
        Y = self.data.interaction_mat
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10 = Y.dot(Y.T) - A8 - A9
        # addition and row-normalization
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
        H_j = sum([A8,A9])
        H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p>3)
        H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
        return [H_s, H_j, H_p]

    def build(self):
        self.n_channel = 4
        # Initialize user and item embeddings
        self.user_embeddings = nn.Parameter(torch.FloatTensor(self.data.user_num, self.emb_size))
        self.item_embeddings = nn.Parameter(torch.FloatTensor(self.data.item_num, self.emb_size))
        nn.init.xavier_uniform_(self.user_embeddings)
        nn.init.xavier_uniform_(self.item_embeddings)        
        # Define learnable parameters
        self.gating_weights = nn.ParameterDict()
        self.gating_bias = nn.ParameterDict()
        self.sgating_weights = nn.ParameterDict()
        self.sgating_bias = nn.ParameterDict()
        for i in range(self.n_channel):
            channel = str(i + 1)
            self.gating_weights[channel] = nn.Parameter(torch.FloatTensor(self.emb_size, self.emb_size))
            self.gating_bias[channel] = nn.Parameter(torch.FloatTensor(1, self.emb_size))
            self.sgating_weights[channel] = nn.Parameter(torch.FloatTensor(self.emb_size, self.emb_size))
            self.sgating_bias[channel] = nn.Parameter(torch.FloatTensor(1, self.emb_size))
            nn.init.xavier_uniform_(self.gating_weights[channel])
            nn.init.zeros_(self.gating_bias[channel])
            nn.init.xavier_uniform_(self.sgating_weights[channel])
            nn.init.zeros_(self.sgating_bias[channel])
        self.attention = nn.Parameter(torch.FloatTensor(1, self.emb_size))
        self.attention_mat = nn.Parameter(torch.FloatTensor(self.emb_size, self.emb_size))
        nn.init.xavier_uniform_(self.attention)
        nn.init.xavier_uniform_(self.attention_mat)
        # Initialize adjacency matrices
        M_matrices = self.build_hyper_adj_mats()
        self.H_s = self.sparse_mx_to_torch_sparse_tensor(M_matrices[0])
        self.H_j = self.sparse_mx_to_torch_sparse_tensor(M_matrices[1])
        self.H_p = self.sparse_mx_to_torch_sparse_tensor(M_matrices[2])
        normalized_R = Graph.normalize_graph_mat(self.data.interaction_mat)
        self.R = self.sparse_mx_to_torch_sparse_tensor(normalized_R)

    def self_gating(self, em, channel):
        channel_str = str(channel)
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weights[channel_str]) + self.gating_bias[channel_str]))

    def self_supervised_gating(self, em, channel):
        channel_str = str(channel)
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.sgating_weights[channel_str]) + self.sgating_bias[channel_str]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(torch.sum(torch.multiply(self.attention, torch.matmul(embedding, self.attention_mat)), 1))
        score = F.softmax(torch.stack(weights), dim=0)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.mul(score[i].view(-1, 1), channel_embeddings[i])
        return mixed_embeddings, score

    def forward(self, u_idx, v_idx, neg_idx):
        user_embeddings_c1 = self.self_gating(self.user_embeddings, 1)
        user_embeddings_c2 = self.self_gating(self.user_embeddings, 2)
        user_embeddings_c3 = self.self_gating(self.user_embeddings, 3)
        simple_user_embeddings = self.self_gating(self.user_embeddings, 4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings
        all_embeddings_i = [item_embeddings]
        ss_loss = 0
        # Multi-channel convolution
        for k in range(self.n_layers):
            # Calculate mixed embeddings using attention
            mixed_embedding, _ = self.channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
            mixed_embedding = mixed_embedding + simple_user_embeddings / 2
            # Channel S
            user_embeddings_c1 = torch.sparse.mm(self.H_s, user_embeddings_c1)
            norm_embeddings_c1 = F.normalize(user_embeddings_c1, p=2, dim=1)
            all_embeddings_c1.append(norm_embeddings_c1)
            # Channel J
            user_embeddings_c2 = torch.sparse.mm(self.H_j, user_embeddings_c2)
            norm_embeddings_c2 = F.normalize(user_embeddings_c2, p=2, dim=1)
            all_embeddings_c2.append(norm_embeddings_c2)
            # Channel P
            user_embeddings_c3 = torch.sparse.mm(self.H_p, user_embeddings_c3)
            norm_embeddings_c3 = F.normalize(user_embeddings_c3, p=2, dim=1)
            all_embeddings_c3.append(norm_embeddings_c3)
            # Item convolution
            new_item_embeddings = torch.sparse.mm(self.R.transpose(0, 1), mixed_embedding)
            norm_embeddings_i = F.normalize(new_item_embeddings, p=2, dim=1)
            all_embeddings_i.append(norm_embeddings_i)
            # Simple user embeddings
            simple_user_embeddings = torch.sparse.mm(self.R, item_embeddings)
            all_embeddings_simple.append(F.normalize(simple_user_embeddings, p=2, dim=1))
            item_embeddings = new_item_embeddings
        # Averaging the channel-specific embeddings
        user_embeddings_c1 = torch.stack(all_embeddings_c1).sum(dim=0)
        user_embeddings_c2 = torch.stack(all_embeddings_c2).sum(dim=0)
        user_embeddings_c3 = torch.stack(all_embeddings_c3).sum(dim=0)
        simple_user_embeddings = torch.stack(all_embeddings_simple).sum(dim=0)
        item_embeddings = torch.stack(all_embeddings_i).sum(dim=0)
        # Aggregating channel-specific embeddings
        final_item_embeddings = item_embeddings
        final_user_embeddings, attention_score = self.channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        final_user_embeddings = final_user_embeddings + simple_user_embeddings / 2
        # Create self-supervised loss
        ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(final_user_embeddings, 1), self.H_s)
        ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(final_user_embeddings, 2), self.H_j)
        ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(final_user_embeddings, 3), self.H_p)
        ss_loss = self.ss_rate * ss_loss
        # Embedding look-up
        batch_user_emb = final_user_embeddings[u_idx]
        batch_pos_item_emb = final_item_embeddings[v_idx]
        batch_neg_item_emb = final_item_embeddings[neg_idx]
        return batch_user_emb, batch_pos_item_emb, batch_neg_item_emb, ss_loss, final_user_embeddings, final_item_embeddings

    def hierarchical_self_supervision(self, em, adj):
        def row_shuffle(embedding):
            indices = torch.randperm(embedding.size(0))
            return embedding[indices]
        
        def row_column_shuffle(embedding):
            indices_row = torch.randperm(embedding.size(0))
            corrupted_embedding = embedding[indices_row]
            return corrupted_embedding
        
        def score(x1, x2):
            return torch.sum(torch.multiply(x1, x2), 1)

        user_embeddings = em
        edge_embeddings = torch.sparse.mm(adj, user_embeddings)        
        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))
        # Global MIM
        graph = torch.mean(edge_embeddings, 0, keepdim=True)
        pos = score(edge_embeddings, graph.expand_as(edge_embeddings))
        neg1 = score(row_column_shuffle(edge_embeddings), graph.expand_as(edge_embeddings))
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    # Override the parent's train method to avoid conflicts
    def train_model(self):
        self.build()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            self.train_epoch(optimizer, epoch)
            with torch.no_grad():
                self.U, self.V = self.final_user_embeddings.detach().cpu().numpy(), self.final_item_embeddings.detach().cpu().numpy()
                self.fast_evaluation(epoch)
        self.save()
        self.U, self.V = self.best_user_emb, self.best_item_emb
        return self.evaluate()

    train = train_model
    def train_epoch(self, optimizer, epoch):
        super(MHCN, self).train()
        for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
            user_idx, i_idx, j_idx = batch
            user_idx = torch.LongTensor(user_idx)
            i_idx = torch.LongTensor(i_idx)
            j_idx = torch.LongTensor(j_idx)
            batch_user_emb, batch_pos_item_emb, batch_neg_item_emb, ss_loss, self.final_user_embeddings, self.final_item_embeddings = self.forward(user_idx, i_idx, j_idx)
            # Calculate loss
            rec_loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
            reg_loss = 0
            # L2 regularization
            for name, param in self.named_parameters():
                reg_loss += self.reg * torch.norm(param, 2)
            total_loss = rec_loss + reg_loss + ss_loss
            # Update parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if (n + 1) % 100 == 0:
                print(f'Training: {epoch + 1} Batch: {n + 1} Rec loss: {rec_loss.item():.4f} SSL loss: {ss_loss.item():.4f}')

    def save(self):
        self.best_user_emb, self.best_item_emb = self.final_user_embeddings.detach().cpu().numpy(), self.final_item_embeddings.detach().cpu().numpy()

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(torch.from_numpy(self.V), torch.from_numpy(self.U[u])).cpu().numpy()
    
    def evaluate(self):
        rec_list = self.test()
        metrics = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print("Detailed TopN Evaluation:")
        print("".join(metrics))
        return {k: float(v) for m in metrics[1:] if ':' in m for k, v in [m.split(':', 1)]}
    

# ---------------------------- Hyperparameter Tuner ----------------------------
class MHCNTuner:
    def __init__(self, train_set, test_set, social_data, base_config):
        self.train_set, self.test_set, self.social_data = train_set, test_set, social_data
        self.base = base_config
        self.results = []
        self.grid = {
            'emb_size': [16, 32, 64, 128, 256, 512],
            'batch_size': [128, 256, 512, 1024, 2048, 4096],
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'reg_lambda': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'MHCN.n_layer': [1, 2, 3, 4],
            'MHCN.ss_rate': [0.001, 0.005, 0.01, 0.05]
        }
        self.default = {
            'emb_size': 64,
            'batch_size': 2048,
            'lr': 0.001,
            'reg_lambda':0.0001,
            'MHCN.n_layer': 2,
            'MHCN.ss_rate': 0.01
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
                    model = MHCN(conf, self.train_set, self.test_set, self.social_data)
                    metrics = model.train()
                    print(f"Result: NDCG: {metrics.get('NDCG')}, Recall: {metrics.get('Recall')}, "
                          f"Precision: {metrics.get('Precision')}, Hit Ratio: {metrics.get('Hit Ratio')}")
                    self.results.append({'config': conf, 'metrics': metrics})
                except Exception as e:
                    import traceback
                    print(f"Error tuning {key} = {val}: {e}")
                    traceback.print_exc()
        with open('mhcn_tuning_individual.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'emb_size': params['emb_size'],
            'batch_size': params['batch_size'],
            'lr': params['lr'],
            'reg_lambda': params['reg_lambda'],
            'max.epoch': 5,
            'item.ranking.topN': [10, 20, 30, 50],
            'MHCN': {
                'n_layer': params['MHCN.n_layer'],
                'ss_rate': params['MHCN.ss_rate']
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
        'model': {'name': 'MHCN', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    social_data = load_data(base_config['social.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print(f"Loaded {len(social_data)} social interactions")
    print("\nMHCN Hyperparameter Tuning Framework\n" + "="*80)
    tuner = MHCNTuner(train_set, test_set, social_data, base_config)
    tuner.run()