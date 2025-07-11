import os
import yaml
import itertools
import json
import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
from random import shuffle, randint, choice, sample
from time import strftime, localtime, time
import math
from os.path import abspath
from re import split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Configuration Class
class ModelConf(object):
    def __init__(self, config_dict=None, file=None):
        self.config = {}
        if file:
            self.read_configuration(file)
        elif config_dict:
            self.config = config_dict

    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def __getitem__(self, item):
        if not self.contain(item):
            print('Parameter ' + item + ' is not found in the configuration file!')
            exit(-1)
        return self.config[item]

    def contain(self, key):
        return key in self.config

    def read_configuration(self, file):
        if not os.path.exists(file):
            print('Config file is not found!')
            raise IOError
        with open(file, 'r') as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error in configuration file: {exc}")
                raise IOError     
        self.optimizer_type = self.config.get('optimizer', 'adam')


# Logger Class
class Log(object):
    def __init__(self, name, log_id):
        self.name = name
        self.log_id = log_id
        self.log_content = []

    def add(self, content):
        self.log_content.append(content + '\n')

    def save(self, file_path):
        with open(file_path, 'w') as f:
            f.writelines(self.log_content)


# File I/O Class
class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def load_data_set(file, rec_type):
        if rec_type == 'graph':
            data = []
            with open(file) as f:
                for line in f:
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                    data.append([user_id, item_id, float(weight)])
        return data


# Evaluation Metrics
class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N),5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure


# Utility Functions
def find_k_largest(k, candidates):
    n_candidates = []
    for iid, score in candidates.items():
        n_candidates.append((score, iid))
    n_candidates.sort(reverse=True)
    k_largest = n_candidates[:k]
    ids = [item[1] for item in k_largest]
    scores = [item[0] for item in k_largest]
    return ids, scores


def next_batch_pairwise(data, batch_size, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


# Torch Interface
class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        coords = np.array([coo.row, coo.col])
        i = torch.LongTensor(coords)
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape).to(device)


# Data Classes
class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training
        self.test_data = test


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


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for user, item, rating in self.training_data:
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id
                self.id2user[user_id] = user
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        for user, item, rating in self.test_data:
            if user in self.user and item in self.item:
                self.test_set[user][item] = rating
                self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())


# Base Recommender Classes
class Recommender:
    def __init__(self, conf, training_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)

        model_config = self.config['model']
        self.model_name = model_config['name']
        self.ranking = self.config['item.ranking.topN']
        self.emb_size = int(self.config['embedding.size'])
        self.maxEpoch = int(self.config['max.epoch'])
        self.batch_size = int(self.config['batch.size'])
        self.lRate = float(self.config['learning.rate'])
        self.reg = float(self.config['reg.lambda'])
        self.output = self.config['output']

        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, f"{self.model_name} {current_time}")

        self.result = []
        self.recOutput = []

    def print_model_info(self):
        print('Model:', self.model_name)
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:', self.reg)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            rated_list, _ = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        # performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}
        performance = {}
        for m in measure[1:]:
            if ':' in m:
                parts = m.strip().split(':')
                if len(parts) == 2:
                    k, v = parts
                    performance[k.strip()] = float(v.strip())
        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        print(f'*Current Performance*\nEpoch: {epoch + 1}')
        for m in measure:
            print(m.strip())
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure


# Neural Network Components
class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


class SelfCF_HE(nn.Module):
    def __init__(self, data, emb_size, momentum, n_layers):
        super(SelfCF_HE, self).__init__()
        self.to(device)
        self.user_count = data.user_num
        self.item_count = data.item_num
        self.latent_size = emb_size
        self.momentum = momentum
        self.online_encoder = LGCN_Encoder(data, emb_size, n_layers)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.u_target_his = torch.randn((self.user_count, self.latent_size), requires_grad=False)
        self.i_target_his = torch.randn((self.item_count, self.latent_size), requires_grad=False)

    def forward(self, inputs):
        u_online, i_online = self.online_encoder()
        with torch.no_grad():
            users = torch.LongTensor(inputs['user']).to(device)
            items = torch.LongTensor(inputs['item']).to(device)
            u_target, i_target = self.u_target_his.clone()[users], self.i_target_his.clone()[items]
            u_target.detach()
            i_target.detach()
            u_target = u_target * self.momentum + u_online[users].data * (1. - self.momentum)
            i_target = i_target * self.momentum + i_online[items].data * (1. - self.momentum)
            self.u_target_his[users, :] = u_online[users].clone()
            self.i_target_his[items, :] = i_online[items].clone()
        return self.predictor(u_online[users]), u_target, self.predictor(i_online[items]), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.forward()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):
        return 1 - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output
        loss_ui = self.loss_fn(u_online, i_target)/2
        loss_iu = self.loss_fn(i_online, u_target)/2
        return loss_ui + loss_iu


# SelfCF Model
class SelfCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SelfCF, self).__init__(conf, training_set, test_set)
        args = self.config['SelfCF']
        self.momentum = float(args['tau'])
        self.n_layers = int(args['n_layer'])
        self.optimizer_type = self.config.get('optimizer', 'adam').lower()
        self.reg_weight = float(self.config.get('reg.weight', 1.0))
        self.model = SelfCF_HE(self.data, self.emb_size, self.momentum, self.n_layers)

    def train(self):
        model = self.model.to(device)
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lRate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        for epoch in range(self.maxEpoch):
            epoch_loss = 0
            batch_count = 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, i_idx, j_idx = batch
                inputs = {'user': user_idx, 'item': i_idx}
                inputs = {k: torch.LongTensor(v).to(device) for k, v in inputs.items()}
                model.train()
                output = model(inputs)
                # batch_loss = model.get_loss(output)
                batch_loss = self.reg_weight * model.get_loss(output)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                batch_count += 1
                if n % 100 == 0:
                    print(f'Epoch {epoch + 1}, Batch {n}, Loss: {batch_loss.item():.4f}')
            
            avg_loss = epoch_loss / batch_count
            print(f'Epoch {epoch + 1} Average Loss: {avg_loss:.4f}')
            
            model.eval()
            with torch.no_grad():
                self.p_u_online, self.u_online, self.p_i_online, self.i_online = self.model.get_embedding()
            
            if epoch % 2 == 0:  # Evaluate every 2 epochs
                self.fast_evaluation(epoch)
        
        self.p_u_online, self.u_online, self.p_i_online, self.i_online = self.best_p_u, self.best_u, self.best_p_i, self.best_i

    def save(self):
        with torch.no_grad():
            self.best_p_u, self.best_u, self.best_p_i, self.best_i = [x.to(device) for x in self.model.get_embedding()]

    # def predict(self, u):
    #     u = self.data.get_user_id(u)
    #     if u is None:
    #         return {}
    #     u_tensor = torch.LongTensor([u]).to(device)
    #     score_ui = torch.matmul(self.p_u_online[u_tensor], self.i_online.transpose(0, 1))
    #     score_iu = torch.matmul(self.u_online[u_tensor], self.p_i_online.transpose(0, 1))
    #     score = score_ui + score_iu
    #     scores_dict = {}
    #     for i in range(len(score)):
    #         scores_dict[i] = score[i].item()
    #     return scores_dict


    def predict(self, u):
        u = self.data.get_user_id(u)
        if u is None:
            return {}
        u_tensor = torch.LongTensor([u]).to(device)
        score_ui = torch.matmul(self.p_u_online[u_tensor], self.i_online.transpose(0, 1))
        score_iu = torch.matmul(self.u_online[u_tensor], self.p_i_online.transpose(0, 1))
        score = score_ui + score_iu
        score = score.squeeze(0)
        scores_dict = {i: score[i].item() for i in range(len(score))}
        # scores_dict = {i: s for i, s in enumerate(score.detach().cpu().numpy().tolist())}
        return scores_dict



# Hyperparameter Tuning Class
class SelfCFTuner:
    def __init__(self, train_set, test_set, base_config):
        self.train_set, self.test_set = train_set, test_set
        self.base = base_config
        self.results = []
        self.grid = {
            'embedding.size': [32, 64, 128, 256, 512, 1024],
            'batch.size': [128, 256, 512, 1024, 2048, 4096],
            'learning.rate': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.2],
            'reg.lambda': [1e-4, 1e-3, 1e-2],
            'reg.weight': [1e-5, 1e-4, 1e-3],
            'SelfCF.tau': [0.07, 0.1, 0.2],
            'SelfCF.n_layer': [1, 2, 3, 4]
        }
        self.default = {
            'embedding.size': 128,
            'batch.size': 1024,
            'learning.rate': 1e-3,
            'reg.lambda': 1e-4,
            'reg.weight': 1.0,
            'SelfCF.tau': 0.2,
            'SelfCF.n_layer': 2
        }

    def run(self):
        total_runs = sum(len(v) for v in self.grid.values())
        print(f"Total combinations: {total_runs}\n" + '='*80)
        run_count = 0
        for key, values in self.grid.items():
            print(f"\n{'='*80}\nTuning hyperparameter: {key}")
            for val in values:
                run_count += 1
                print(f"\n>>> [{run_count}\{total_runs}] {key} = {val}")
                param_config = self.default.copy()
                param_config[key] = val
                conf = self.make_config(param_config)
                try:
                    # print(f"\n>>> {key} = {val}")
                    model_conf = ModelConf(config_dict=conf)
                    model = SelfCF(model_conf, self.train_set, self.test_set)
                    model.execute()
                    if model.bestPerformance:
                        metrics = model.bestPerformance[1]
                    else:
                        metrics = {}
                    print(f"Result: Epoch {model.bestPerformance[0] if model.bestPerformance else 0}, {metrics}")
                    self.results.append({'param': key, 'value': val, 'metrics': metrics})
                except Exception as e:
                    import traceback
                    print(f"Error tuning {key} = {val}: {e}")
                    traceback.print_exc()
        with open('selfcf_tuning_individual.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'embedding.size': params['embedding.size'],
            'batch.size': params['batch.size'],
            'learning.rate': params['learning.rate'],
            'reg.lambda': params['reg.lambda'],
            'reg.weight': params['reg.weight'],
            'max.epoch': 1,
            'item.ranking.topN': [10, 20, 30, 50],
            'model': {'name': 'SelfCF', 'type': 'graph'},
            'output': './output/'
        })
        conf['SelfCF'] = {
            'tau': params['SelfCF.tau'],
            'n_layer': params['SelfCF.n_layer']
        }
        return conf

    
if __name__ == '__main__':
    train_path = './dataset/ml100k/train.txt'
    test_path = './dataset/ml100k/test.txt'
    print("Loading data from configuration files...")
    train_data = FileIO.load_data_set(train_path, rec_type='graph')
    test_data = FileIO.load_data_set(test_path, rec_type='graph')
    print(f"Loaded {len(train_data)} training interactions")
    print(f"Loaded {len(test_data)} test interactions")
    print("\nSelfCF Hyperparameter Tuning Framework\n" + "="*80)
    base_conf = {}
    tuner = SelfCFTuner(train_data, test_data, base_conf)
    tuner.run()