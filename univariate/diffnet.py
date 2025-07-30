import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
import numpy as np
import os
import itertools
import traceback
from collections import defaultdict
import os, math, heapq, random, json, copy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import scipy.sparse as sp
from random import shuffle,randint,choice
from numpy.linalg import norm
from math import sqrt,exp
from numba import jit
from os.path import abspath
from time import strftime,localtime,time
import os.path
from os import makedirs,remove
from re import compile,findall,split


class Rating(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.userMeans = {}
        self.itemMeans = {}
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)
        self.rScale = []
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.__generateSet()
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()
        if self.evalSettings.contains('-cold'):
            self.__cold_start_test()

    def __generateSet(self):
        scale = set()
        if self.evalSettings.contains('-val'):
            random.shuffle(self.trainingData)
            separation = int(self.elemCount()*float(self.evalSettings['-val']))
            self.testData = self.trainingData[:separation]
            self.trainingData = self.trainingData[separation:]
        for i,entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            if userName not in self.user:
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if itemName not in self.item:
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            if self.evalSettings.contains('-predict'):
                self.testSet_u[entry]={}
            else:
                userName, itemName, rating = entry
                self.testSet_u[userName][itemName] = rating
                self.testSet_i[itemName][userName] = rating

    def __cold_start_test(self):
        #evaluation on cold-start users
        threshold = int(self.evalSettings['-cold'])
        removedUser = {}
        for user in self.testSet_u:
            if user in self.trainSet_u and len(self.trainSet_u[user])>threshold:
                removedUser[user]=1
        for user in removedUser:
            del self.testSet_u[user]
        testData = []
        for item in self.testData:
            if item[0] not in removedUser:
                testData.append(item)
        self.testData = testData

    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            self.userMeans[u] = sum(self.trainSet_u[u].values())/len(self.trainSet_u[u])

    def __computeItemMean(self):
        for c in self.item:
            self.itemMeans[c] = sum(self.trainSet_i[c].values())/len(self.trainSet_i[c])

    def getUserId(self,u):
        if u in self.user:
            return self.user[u]

    def getItemId(self,i):
        if i in self.item:
            return self.item[i]

    def trainingSize(self):
        return (len(self.user),len(self.item),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether user u rated item i'
        if u in self.user and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsUser(self,u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def containsItem(self,i):
        'whether item is in training set'
        if i in self.item:
            return True
        else:
            return False

    def userRated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def itemRated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())

    def row(self,u):
        k,v = self.userRated(u)
        vec = np.zeros(len(self.item))
        for pair in zip(k,v):
            iid = self.item[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.itemRated(i)
        vec = np.zeros(len(self.user))
        for pair in zip(k,v):
            uid = self.user[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user),len(self.item)))
        for u in self.user:
            k, v = self.userRated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]]=vec
        return m

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)



class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = Rating(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = OptionConf(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.num_users, self.num_items, self.train_size = self.data.trainingSize()

    def readConfiguration(self):
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = OptionConf(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:',self.config['model.name'])
        print('Ratings dataset:',abspath(self.config['ratings']))
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        #print dataset statistics
        print('Training set size: (user count: %d, item count %d, record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (user count: %d, item count %d, record count: %d)' %(self.data.testSize()))
        print('='*80)
        #print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr+=key[1:]+':'+args[key]+'  '
            print('Specific parameters:',parStr)
            print('=' * 80)

    def initModel(self):
        pass

    def trainModel(self):
        'build the model (for model-based Models )'
        pass

    def trainModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predictForRating(self, u, i):
        pass

    def predictForRanking(self,u):
        pass

    def checkRatingBoundary(self,prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction,3)

    def evalRatings(self):
        res = list()
        res.append('userId  itemId  original  prediction\n')
        for ind,entry in enumerate(self.data.testData):
            user,item,rating = entry
            prediction = self.predictForRating(user, item)
            pred = self.checkRatingBoundary(prediction)
            self.data.testData[ind].append(pred)
            res.append(user+' '+item+' '+str(rating)+' '+str(pred)+'\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S",localtime(time()))
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['model.name']+'@'+currentTime+'-rating-predictions'+self.foldInfo+'.txt'
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.data.testData)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        print('The result of %s %s:\n%s' % (self.modelName, self.foldInfo, ''.join(self.measure)))

    def evalRanking(self):
        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = max(top)
            if N > 100 or N < 1:
                print('N can not be larger than 100! It has been reassigned to 10')
                N = 10
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userCount = len(self.data.testSet_u)
        #rawRes = {}
        for i, user in enumerate(self.data.testSet_u):
            line = user + ':'
            candidates = self.predictForRanking(user)
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                candidates[self.data.item[item]] = 0
            ids,scores = find_k_largest(N,candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names,scores))
            if i % 100 == 0:
                print(self.modelName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount))
            for item in recList[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.testSet_u[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['model.name'] + '@' + currentTime + '-top-' + str(
            N) + 'items' + self.foldInfo + '.txt'
        # output evaluation result
        if self.evalSettings.contains('-predict'):
            exit(0)
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        
    def execute(self):
        self.readConfiguration()
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' %self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' %self.foldInfo)
            self.initModel()
            print('Building Model %s...' %self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.trainModel_tf()
                else:
                    self.trainModel()
            except ImportError:
                self.trainModel()
        print('Predicting %s...' %self.foldInfo)
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()
        if self.isSaveModel:
            print('Saving model %s...' %self.foldInfo)
            self.saveModel()
        return self.measure


class sparseMatrix():
    'matrix used to store raw data'
    def __init__(self,triple):
        self.matrix_User = {}
        self.matrix_Item = {}
        for item in triple:
            if item[0] not in self.matrix_User:
                self.matrix_User[item[0]] = {}
            if item[1] not in self.matrix_Item:
                self.matrix_Item[item[1]] = {}
            self.matrix_User[item[0]][item[1]]=item[2]
            self.matrix_Item[item[1]][item[0]]=item[2]
        self.elemNum = len(triple)
        self.size = (len(self.matrix_User),len(self.matrix_Item))

    def sRow(self,r):
        if r not in self.matrix_User:
            return {}
        else:
            return self.matrix_User[r]

    def sCol(self,c):
        if c not in self.matrix_Item:
            return {}
        else:
            return self.matrix_Item[c]

    def row(self,r):
        if r not in self.matrix_User:
            return np.zeros((1,self.size[1]))
        else:
            array = np.zeros((1,self.size[1]))
            ind = list(self.matrix_User[r].keys())
            val = list(self.matrix_User[r].values())
            array[0][ind] = val
            return array

    def col(self,c):
        if c not in self.matrix_Item:
            return np.zeros((1,self.size[0]))
        else:
            array = np.zeros((1,self.size[0]))
            ind = list(self.matrix_Item[c].keys())
            val = list(self.matrix_Item[c].values())
            array[0][ind] = val
            return array
    def elem(self,r,c):
        if not self.contains(r,c):
            return 0
        return self.matrix_User[r][c]

    def contains(self,r,c):
        if r in self.matrix_User and c in self.matrix_User[r]:
            return True
        return False

    def elemCount(self):
        return self.elemNum

    def size(self):
        return self.size


class Social(object):
    def __init__(self,conf,relation=None):
        self.config = conf
        self.user = {}
        self.relation = relation
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self.trustMatrix = self.__generateSet()

    def __generateSet(self):
        triple = []
        for line in self.relation:
            userId1,userId2,weight = line
            #add relations to dict
            self.followees[userId1][userId2] = weight
            self.followers[userId2][userId1] = weight
            # order the user
            if userId1 not in self.user:
                self.user[userId1] = len(self.user)
            if userId2 not in self.user:
                self.user[userId2] = len(self.user)
            triple.append([self.user[userId1], self.user[userId2], weight])
        return sparseMatrix(triple)

    def row(self,u):
        return self.trustMatrix.row(self.user[u])

    def col(self,u):
        return self.trustMatrix.col(self.user[u])

    def elem(self,u1,u2):
        return self.trustMatrix.elem(u1,u2)

    def weight(self,u1,u2):
        if u1 in self.followees and u2 in self.followees[u1]:
            return self.followees[u1][u2]
        else:
            return 0

    def trustSize(self):
        return self.trustMatrix.size

    def getFollowers(self,u):
        if u in self.followers:
            return self.followers[u]
        else:
            return {}

    def getFollowees(self,u):
        if u in self.followees:
            return self.followees[u]
        else:
            return {}

    def hasFollowee(self,u1,u2):
        if u1 in self.followees:
            if u2 in self.followees[u1]:
                return True
            else:
                return False
        return False

    def hasFollower(self,u1,u2):
        if u1 in self.followers:
            if u2 in self.followers[u1]:
                return True
            else:
                return False
        return False


class OptionConf(object):
    def __init__(self,content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i,item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1

    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.options[item]

    def keys(self):
        return self.options.keys()

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return key in self.options
    

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


class Measure(object):
    def __init__(self):
        pass
    @staticmethod
    def ratingMeasure(res):
        measure = []
        mae = Measure.MAE(res)
        measure.append('MAE:'+str(mae)+'\n')
        rmse = Measure.RMSE(res)
        measure.append('RMSE:' + str(rmse)+'\n')
        return measure

    @staticmethod
    def hits(origin, res):
        hitCount = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hitCount[user] = len(set(items).intersection(set(predicted)))
        return hitCount

    @staticmethod
    def rankingMeasure(origin, res, N):
        measure = []
        for n in N:
            predicted = {}
            for user in res:
                predicted[user] = res[user][:n]
            indicators = []
            if len(origin) != len(predicted):
                print('The Lengths of test set and predicted set are not match!')
                exit(-1)
            hits = Measure.hits(origin, predicted)
            prec = Measure.precision(hits, n)
            indicators.append('Precision:' + str(prec) + '\n')
            recall = Measure.recall(hits, origin)
            indicators.append('Recall:' + str(recall) + '\n')
            F1 = Measure.F1(prec, recall)
            indicators.append('F1:' + str(F1) + '\n')
            NDCG = Measure.NDCG(origin, predicted, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            measure.append('Top ' + str(n) + '\n')
            measure += indicators
        return measure

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

    @staticmethod
    def recall(hits, origin):
        recallList = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recallList) / len(recallList)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)
    

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


class IterativeRecommender(Recommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(IterativeRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.bestPerformance = []
        self.earlyStop = 0

    def readConfiguration(self):
        super(IterativeRecommender, self).readConfiguration()
        self.emb_size = int(self.config['emb_size'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        learningRate = OptionConf(self.config['lr'])
        self.lRate = float(self.config['lr'])
        self.maxLRate = float(learningRate['-max'])
        if self.evalSettings.contains('-tf'):
            self.batch_size = int(self.config['batch_size'])
        regular = OptionConf(self.config['reg_lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])

    def printAlgorConfig(self):
        super(IterativeRecommender, self).printAlgorConfig()
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Regularization parameter: regU %.3f, regI %.3f, regB %.3f' %(self.regU,self.regI,self.regB))
        print('='*80)

    def initModel(self):
        self.P = np.random.rand(len(self.data.user), self.emb_size) / 3
        self.Q = np.random.rand(len(self.data.item), self.emb_size) / 3
        self.loss, self.lastLoss = 0, 0

    def trainModel_tf(self):
        self.U = nn.Parameter(torch.randn(self.num_users, self.emb_size) * 0.005)
        self.V = nn.Parameter(torch.randn(self.num_items, self.emb_size) * 0.005)
        self.user_biases = nn.Parameter(torch.randn(self.num_users, 1) * 0.005)
        self.item_biases = nn.Parameter(torch.randn(self.num_items, 1) * 0.005)        
        self.u_idx = None
        self.v_idx = None
        self.r = None        
        self.user_embedding = None
        self.item_embedding = None
        self.user_bias = None
        self.item_bias = None
        def forward(self, u_idx, v_idx, r=None):
            self.u_idx = torch.tensor(u_idx, dtype=torch.long)
            self.v_idx = torch.tensor(v_idx, dtype=torch.long)
            if r is not None:
                self.r = torch.tensor(r, dtype=torch.float)                
            self.user_bias = self.user_biases[self.u_idx]
            self.item_bias = self.item_biases[self.v_idx]
            self.user_embedding = self.U[self.u_idx]
            self.item_embedding = self.V[self.v_idx]

    def updateLearningRate(self,epoch):
        if epoch > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.lRate *= 1.05
            else:
                self.lRate *= 0.5
        if self.lRate > self.maxLRate > 0:
            self.lRate = self.maxLRate

    def predictForRating(self, u, i):
        if self.data.containsUser(u) and self.data.containsItem(i):
            return self.P[self.data.user[u]].dot(self.Q[self.data.item[i]])
        elif self.data.containsUser(u) and not self.data.containsItem(i):
            return self.data.userMeans[u]
        elif not self.data.containsUser(u) and self.data.containsItem(i):
            return self.data.itemMeans[i]
        else:
            return self.data.globalMean

    def predictForRanking(self,u):
        'used to rank all the items for the user'
        if self.data.containsUser(u):
            return self.Q.dot(self.P[self.data.user[u]])
        else:
            return [self.data.globalMean]*self.num_items

    def isConverged(self,epoch):
        from math import isnan
        if isnan(self.loss):
            print('Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            print('%s %s epoch %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f' \
                  % (self.modelName, self.foldInfo, epoch, self.loss, deltaLoss, self.lRate))
        else:
            measure = self.rating_performance()
            print('%s %s epoch %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
                  % (self.modelName, self.foldInfo, epoch, self.loss, deltaLoss, self.lRate, measure[0].strip()[:11], measure[1].strip()[:12]))
        #check if converged
        cond = abs(deltaLoss) < 1e-3
        converged = cond
        if not converged:
            self.updateLearningRate(epoch)
        self.lastLoss = self.loss
        shuffle(self.data.trainingData)
        return converged

    def rating_performance(self):
        res = []
        for ind, entry in enumerate(self.data.testData):
            user, item, rating = entry
            # predict
            prediction = self.predictForRating(user, item)
            pred = self.checkRatingBoundary(prediction)
            res.append([user,item,rating,pred])
        self.measure = Measure.ratingMeasure(res)
        return self.measure

    def ranking_performance(self,epoch):
        #evaluation during training
        top = self.conf.get('item.ranking.topN', [10, 20, 30, 50])
        top = [int(num) for num in top]
        N = max(top)
        recList = {}
        print('Evaluating...')
        for user in self.data.testSet_u:
            candidates = self.predictForRanking(user)
            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                candidates[self.data.item[item]] = 0
            ids, scores = find_k_largest(N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names, scores))
        measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        if len(self.bestPerformance)>0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k,v = m.strip().split(':')
                performance[k]=float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -=1
            if count<0:
                self.bestPerformance[1]=performance
                self.bestPerformance[0]=epoch+1
                self.saveModel()
        else:
            self.bestPerformance.append(epoch+1)
            performance = {}
            for m in measure[1:]:
                if ':' in m:
                    k,v = m.strip().split(':')
                    performance[k]=float(v)
                    self.bestPerformance.append(performance)
            self.saveModel()
        print('-'*120)
        print('Quick Ranking Performance '+self.foldInfo+' (Top-'+str(N)+'Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch+1))
        for m in measure[1:]:
            print(m.strip())
        bp = ''
        bp += 'Precision'+':'+str(self.bestPerformance[1]['Precision'])+' | '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:',str(self.bestPerformance[0])+',',bp)
        print('-'*120)
        return measure


class SocialRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,relation,fold='[1]'):
        super(SocialRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.social = Social(self.config, relation) #social relations access control
        cleanList = []
        cleanPair = []
        for user in self.social.followees:
            if user not in self.data.user:
                cleanList.append(user)
            for u2 in self.social.followees[user]:
                if u2 not in self.data.user:
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followees[u]
        for pair in cleanPair:
            if pair[0] in self.social.followees:
                del self.social.followees[pair[0]][pair[1]]
        cleanList = []
        cleanPair = []
        for user in self.social.followers:
            if user not in self.data.user:
                cleanList.append(user)
            for u2 in self.social.followers[user]:
                if u2 not in self.data.user:
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followers[u]
        for pair in cleanPair:
            if pair[0] in self.social.followers:
                del self.social.followers[pair[0]][pair[1]]
        idx = []
        for n,pair in enumerate(self.social.relation):
            if pair[0] not in self.data.user or pair[1] not in self.data.user:
                idx.append(n)
        for item in reversed(idx):
            del self.social.relation[item]

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = OptionConf(self.config['reg.lambda'])
        self.regS = float(regular['-s'])

    def printAlgorConfig(self):
        super(SocialRecommender, self).printAlgorConfig()
        print('Social dataset:',abspath(self.config['social']))
        print('Social relation size ','(User count:',len(self.social.user),'Relation count:'+str(len(self.social.relation))+')')
        print('Social Regularization parameter: regS %.3f' % (self.regS))
        print('=' * 80)


class DeepRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(DeepRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def readConfiguration(self):
        super(DeepRecommender, self).readConfiguration()
        self.batch_size = int(self.config['batch_size'])

    def printAlgorConfig(self):
        super(DeepRecommender, self).printAlgorConfig()

    def initModel(self):
        super(DeepRecommender, self).initModel()
        self.user_embeddings = nn.Parameter(torch.randn(self.num_users, self.emb_size) * 0.005).to(self.device)
        self.item_embeddings = nn.Parameter(torch.randn(self.num_items, self.emb_size) * 0.005).to(self.device)        
        self.u_idx = None
        self.v_idx = None
        self.r = None
        self.batch_user_emb = None
        self.batch_pos_item_emb = None

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size
            u_idx, i_idx, j_idx = [], [], []
            item_list = list(self.data.item.keys())
            for i, user in enumerate(users):
                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])
                neg_item = choice(item_list)
                while neg_item in self.data.trainSet_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(self.data.item[neg_item])
            yield u_idx, i_idx, j_idx

    def next_batch_pointwise(self):
        batch_id=0
        while batch_id<self.train_size:
            if batch_id+self.batch_size<=self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
                batch_id+=self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id=self.train_size
            u_idx,i_idx,y = [],[],[]
            for i,user in enumerate(users):
                i_idx.append(self.data.item[items[i]])
                u_idx.append(self.data.user[user])
                y.append(1)
                for instance in range(4):
                    item_j = randint(0, self.num_items - 1)
                    while self.data.id2item[item_j] in self.data.trainSet_u[user]:
                        item_j = randint(0, self.num_items - 1)
                    u_idx.append(self.data.user[user])
                    i_idx.append(item_j)
                    y.append(0)
            yield u_idx,i_idx,y

    def predictForRanking(self,u):
        pass


class GraphRecommender(DeepRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(GraphRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.conf = conf

    def create_joint_sparse_adjaceny(self):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def create_joint_sparse_adj_tensor(self):
        norm_adj = self.create_joint_sparse_adjaceny()
        row,col = norm_adj.nonzero()
        values = norm_adj.data        
        indices = torch.LongTensor(np.vstack([row, col])).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        shape = torch.Size(norm_adj.shape)        
        adj_tensor = torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        return adj_tensor

    def create_sparse_rating_matrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0/len(self.data.trainSet_u[pair[0]])]
        ratingMat = sp.coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_items),dtype=np.float32)
        return ratingMat

    def create_sparse_adj_tensor(self):
        ratingMat = self.create_sparse_rating_matrix()
        row,col = ratingMat.nonzero()
        values = ratingMat.data
        indices = torch.LongTensor(np.vstack([row, col])).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        shape = torch.Size(ratingMat.shape)        
        adj_tensor = torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        return adj_tensor


class DiffNet(SocialRecommender, GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation, fold=fold)
        self.emb_size = conf['emb_size']
        self.batch_size = conf['batch_size']
        self.factors = conf['factors']
        self.lRate = conf['lr']
        self.reg_lambda = conf['reg_lambda']
        self.reg_lambda_u = conf['reg_lambda_u']
        self.reg_lambda_i = conf['reg_lambda_i']
        self.n_layers = conf['n_layer']
        self.regU = float(conf['reg_lambda']) if 'regU' in conf else 1e-4
        self.emb_size = int(conf['emb_size']) if 'emb_size' in conf else 64
        self.user_embeddings = nn.Parameter(torch.randn(self.num_users, self.emb_size) * 0.005).to(self.device)
        self.item_embeddings = nn.Parameter(torch.randn(self.num_items, self.emb_size) * 0.005).to(self.device)  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0/len(self.social.followees[pair[0]])]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_users), dtype=np.float32)
        return AdjacencyMatrix

    def initModel(self):
        super(DiffNet, self).initModel()
        S = self.buildSparseRelationMatrix()        
        indices = torch.LongTensor(np.vstack([S.row, S.col])).to(self.device)
        values = torch.FloatTensor(S.data.astype(np.float32)).to(self.device)
        shape = torch.Size(S.shape)
        self.S = torch.sparse.FloatTensor(indices, values, shape).to(self.device)        
        self.A = self.create_sparse_adj_tensor()        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(2 * self.emb_size, self.emb_size)))
            for _ in range(self.n_layers)
        ]).to(self.device)

    def trainModel(self):
        self.user_embeddings = self.user_embeddings.to(self.device)
        self.item_embeddings = self.item_embeddings.to(self.device)
        self.initModel()
        optimizer = optim.Adam([
            {'params': self.user_embeddings},
            {'params': self.item_embeddings},
            {'params': self.weights}
        ], lr=self.lRate)
        for epoch in range(1):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                # Forward pass
                final_user_embeddings = self.forward()
                u_embedding = final_user_embeddings[user_idx]
                v_embedding = self.item_embeddings[i_idx]
                neg_item_embedding = self.item_embeddings[j_idx]
                # Compute loss
                pos_scores = torch.sum(torch.mul(u_embedding, v_embedding), dim=1)
                neg_scores = torch.sum(torch.mul(u_embedding, neg_item_embedding), dim=1)
                y = pos_scores - neg_scores
                loss = -torch.sum(torch.log(torch.sigmoid(y))) + self.regU * (
                       torch.norm(u_embedding, 2) + torch.norm(v_embedding, 2) +
                       torch.norm(neg_item_embedding, 2))
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (n + 1) % 100 == 0:
                    print('Training', epoch + 1, 'Batch:', n + 1, 'loss:', loss.item())
        return self.ranking_performance(epoch)    

    def forward(self):
        user_embeddings = self.user_embeddings
        for k in range(self.n_layers):
            new_user_embeddings = torch.sparse.mm(self.S, user_embeddings)            
            concat_embeddings = torch.cat([new_user_embeddings, user_embeddings], dim=1)
            user_embeddings = torch.matmul(concat_embeddings, self.weights[k])            
            user_embeddings = torch.relu(user_embeddings)
        final_user_embeddings = user_embeddings + torch.sparse.mm(self.A, self.item_embeddings)
        return final_user_embeddings

    def predictForRanking(self, u):
        if self.data.containsUser(u):
            u_id = self.data.getUserId(u)
            u_idx = torch.LongTensor([u_id]).to(self.device)            
            with torch.no_grad():
                final_user_embeddings = self.forward()
                u_embedding = final_user_embeddings[u_idx]
                scores = torch.sum(torch.mul(u_embedding.unsqueeze(1), self.item_embeddings), dim=2)                
                return scores.squeeze().cpu().numpy()
        else:
            return [self.data.globalMean] * self.num_items
    

# ---------------------------- Load Data ----------------------------
def load_data(path):
    return [[*line.strip().split()[:2], 1.0] for line in open(path) if line.strip()] if os.path.exists(path) else []

# ---------------------------- DiffNetTuner ----------------------------
class DiffNetTuner:
    def __init__(self, train_set, test_set, social_data, base_config):
        self.train_set = train_set
        self.test_set = test_set
        self.social_data = social_data
        self.base = base_config
        self.results = []
        self.grid = {
            'emb_size': [16, 32, 64, 128, 256, 512],
            'batch_size': [128, 256, 512, 1024, 2048, 4096],
            'factors': [10, 20, 30, 40, 50],
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            'reg_lambda': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'reg_lambda_u': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'reg_lambda_i': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'n_layer': [1, 2, 3, 4]
        }
        self.default = {
            'emb_size': 64,
            'batch_size': 2048,
            'factors': 50,
            'lr': 0.001,
            'reg_lambda': 0.0001,
            'reg_lambda_u': 0.001,
            'reg_lambda_i': 0.01,
            'n_layer': 2
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
                    model = DiffNet(conf, self.train_set, self.test_set, self.social_data)
                    metrics = model.trainModel()
                    self.results.append({'config': conf, 'metrics': metrics})
                except Exception as e:
                    print(f"[Error] Tuning {key} = {val}: {e}")
                    traceback.print_exc()
        with open('diffnet_tuning_individual.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def make_config(self, params):
        conf = copy.deepcopy(self.base)
        conf.update({
            'emb_size': params['emb_size'],
            'batch_size': params['batch_size'],
            'factors': params['factors'],
            'lr': params['lr'],
            'reg_lambda': params['reg_lambda'],
            'reg_lambda_u': params['reg_lambda_u'],
            'reg_lambda_i': params['reg_lambda_i'],
            'num.max.epoch': 1,
            'item.ranking.topN': [10, 20, 30, 50],
            'evaluation.setup': 'cv -k 1 -p on -rand-seed 1',
            'n_layer': params['n_layer']
        })
        return conf

# ---------------------------- Main ----------------------------
if __name__ == '__main__':
    print("Loading data from configuration files...")
    base_config = {
        'training.set': './dataset/ml100k/train.txt',
        'test.set': './dataset/ml100k/test.txt',
        'social.set': './dataset/ml100k/social.txt',
        'model': {'name': 'DiffNet', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    social_data = load_data(base_config['social.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print(f"Loaded {len(social_data)} social interactions")
    print("\nDiffNet Hyperparameter Tuning Framework\n" + "="*80)
    tuner = DiffNetTuner(train_set, test_set, social_data, base_config)
    tuner.run()