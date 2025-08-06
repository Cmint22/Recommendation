from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import torch, os, traceback, math, heapq, random, json, copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from collections import defaultdict
from random import shuffle, randint, choice
from os.path import abspath
from time import strftime,localtime,time
from math import sqrt
from numba import jit
import numpy as np
import os
import os.path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



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
                if ':' in m:
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



def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-10
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = torch.log(logits + eps) + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

class ESRF(SocialRecommender, GraphRecommender):
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
        self.reg_lambda_s = conf['reg_lambda_s']
        self.reg_lambda_b = conf['reg_lambda_b']
        self.K = conf['K']
        self.beta = conf['beta']
        self.n_layers = conf['n_layer']
        self.n_layers_G = 2  
        self.n_layers_D = 2
        # self.K = 30
        # self.beta = 0.2
        self.regU = float(conf['reg_lambda']) if 'regU' in conf else 1e-3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bestU = None
        self.bestV = None
        self.bestPerformance = []
        self.U = None
        self.V = None

    def readConfiguration(self):
        super(ESRF, self).readConfiguration()
        args = OptionConf(self.config['ESRF'])
        self.K = int(args['-K'])  # controlling the magnitude of adversarial learning
        self.beta = float(args['-beta'])  # the number of alternative neighbors
        self.n_layers_D = int(args['-n_layer'])  # the number of layers of the recommendation module (discriminator)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_users), dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return ratingMatrix

    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.transpose().tocsr()
        B = S.multiply(S.transpose())
        U = S - B
        C1 = (U.dot(U)).multiply(U.transpose())
        A1 = C1 + C1.transpose()
        C2 = (B.dot(U)).multiply(U.transpose()) + (U.dot(B)).multiply(U.transpose()) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.transpose()
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.transpose()
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(U) + (U.transpose().dot(U)).multiply(U)
        A5 = C5 + C5.transpose()
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.transpose())).multiply(U.transpose()) + (U.transpose().dot(U)).multiply(B)
        A7 = (U.transpose().dot(B)).multiply(U.transpose()) + (B.dot(U)).multiply(U) + (U.dot(U.transpose())).multiply(B)
        self.A8 = (Y.dot(Y.transpose())).multiply(B)
        A9 = (Y.dot(Y.transpose())).multiply(U)
        A10 = Y.dot(Y.transpose())
        for i in range(self.num_users):
            A10[i, i] = 0
        # user pairs which share less than 5 common purchases are ignored
        A10 = A10.multiply(A10 > 5)
        # obtain the normalized high-order adjacency
        A = S + A1 + A2 + A3 + A4 + A5 + A6 + A7 + self.A8 + A9 + A10
        A = A.transpose().multiply(1.0 / A.sum(axis=1).reshape(1, -1))
        A = A.transpose()
        return A

    def convert_sparse_matrix_to_torch_sparse_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    class Generator(nn.Module):
        def __init__(self, num_users, emb_size, n_layers, K):
            super(ESRF.Generator, self).__init__()
            self.relation_embeddings = nn.Parameter(torch.randn(num_users, emb_size) * 0.005)
            self.projection_head = nn.Parameter(torch.randn(emb_size, emb_size) * 0.005)
            self.c_selector = nn.Parameter(torch.randn(K, num_users) * 0.005)
            self.n_layers = n_layers
            self.K = K
            self.num_users = num_users
            self.emb_size = emb_size

        def forward(self, A, user_segment):
            all_embeddings = [self.relation_embeddings]
            user_embeddings = self.relation_embeddings
            for _ in range(self.n_layers):
                user_embeddings = torch.sparse.mm(A, user_embeddings)
                norm_embeddings = F.normalize(user_embeddings, p=2, dim=1)
                all_embeddings.append(norm_embeddings)
            user_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
            # Only process a segment of users (100 at a time to avoid OOM)
            segment_end = min(user_segment + 100, self.num_users)
            user_features = torch.mm(user_embeddings[user_segment:segment_end], user_embeddings.t())
            # Create alternative neighborhood for each user in the segment
            alternative_neighborhood_segment = []
            for i in range(user_features.size(0)):
                embedding = user_features[i]
                alpha_embeddings = embedding.unsqueeze(0) * self.c_selector
                multi_hot_vector = gumbel_softmax(alpha_embeddings, 0.2).sum(dim=0)
                alternative_neighborhood_segment.append(multi_hot_vector)
            alternative_neighborhood_segment = torch.stack(alternative_neighborhood_segment)
            # Create full alternative neighborhood tensor with paddings
            alternative_neighborhood = torch.zeros(self.num_users, self.num_users, device=user_embeddings.device)
            alternative_neighborhood[user_segment:segment_end] = alternative_neighborhood_segment
            return alternative_neighborhood

    class Discriminator(nn.Module):
        def __init__(self, num_users, num_items, emb_size, n_layers):
            super(ESRF.Discriminator, self).__init__()
            self.user_embeddings = nn.Parameter(torch.randn(num_users, emb_size) * 0.01)
            self.item_embeddings = nn.Parameter(torch.randn(num_items, emb_size) * 0.01)
            self.attention_weights = nn.ModuleList()
            for k in range(n_layers):
                self.attention_weights.append(nn.ModuleDict({
                    f'attention_m1{k}': nn.Linear(emb_size, emb_size, bias=False),
                    f'attention_m2{k}': nn.Linear(emb_size, emb_size, bias=False),
                    f'attention_v{k}': nn.Linear(emb_size * 2, 1, bias=False)
                }))
            self.n_layers = n_layers
            self.num_users = num_users
            self.num_items = num_items
            self.emb_size = emb_size

        def forward(self, norm_adj, alternative_neighborhood, is_social, is_attentive, K):
            ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], dim=0)
            all_embeddings = [ego_embeddings]
            for layer in range(self.n_layers):
                new_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
                if is_social:
                    # Calculate social embeddings based on alternative neighborhood
                    nonattentive_social_embeddings = torch.mm(alternative_neighborhood, ego_embeddings[:self.num_users]) / K
                    if is_attentive:
                        # Implement attention mechanism if needed
                        social_embeddings = nonattentive_social_embeddings
                    else:
                        social_embeddings = nonattentive_social_embeddings
                    # Combine with social embeddings
                    user_part = ego_embeddings[:self.num_users] + social_embeddings
                    item_part = ego_embeddings[self.num_users:]
                    ego_embeddings = torch.cat([user_part, item_part], dim=0)
                else:
                    ego_embeddings = new_embeddings
                # Normalize embeddings
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings.append(norm_embeddings)
            all_embeddings = torch.stack(all_embeddings, dim=0).sum(dim=0)
            user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])
            return user_embeddings, item_embeddings

    def initModel(self):
        super(ESRF, self).initModel()
        self.listed_data = []
        for i in range(self.num_users):
            user = self.data.id2user[i]
            items = list(self.data.trainSet_u[user].keys())
            items = [self.data.item[item] for item in items]
            self.listed_data.append(items)
        # Create PyTorch models
        self.generator = self.Generator(self.num_users, self.emb_size, self.n_layers_G, self.K)
        self.discriminator = self.Discriminator(self.num_users, self.num_items, self.emb_size, self.n_layers_D)
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lRate * 5)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lRate)
        # Initialize embeddings with zeros to prevent predictForRanking errors
        self.U = np.zeros((self.num_users, self.emb_size))
        self.V = np.zeros((self.num_items, self.emb_size))

    def trainModel(self):
        # Build adjacency matrices
        A = self.buildMotifInducedAdjacencyMatrix()
        A_tensor = self.convert_sparse_matrix_to_torch_sparse_tensor(A)
        norm_adj = self.create_joint_sparse_adj_tensor()        
        self.attentiveTraining = 0
        self.bestPerformance = []
        self.initModel()
        print('pretraining...')
        for epoch in range(1):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                u_i = np.random.randint(0, self.num_users)
                # Convert numpy arrays to PyTorch tensors
                user_idx_tensor = torch.LongTensor(user_idx)
                i_idx_tensor = torch.LongTensor(i_idx)
                j_idx_tensor = torch.LongTensor(j_idx)
                # Generate empty alternative neighborhood during pretraining
                alternative_neighborhood = torch.zeros(self.num_users, self.num_users)
                # Forward pass through discriminator
                user_emb, item_emb = self.discriminator(norm_adj, alternative_neighborhood, False, self.attentiveTraining, self.K)
                # Get embeddings for specific users and items
                u_emb = user_emb[user_idx_tensor]
                v_emb = item_emb[i_idx_tensor]
                neg_item_emb = item_emb[j_idx_tensor]
                # Calculate loss
                y_ui = torch.sum(u_emb * v_emb, dim=1)
                y_uj = torch.sum(u_emb * neg_item_emb, dim=1)
                pairwise_loss = -torch.sum(torch.log(torch.sigmoid(y_ui - y_uj) + 1e-10))
                reg_loss = self.regU * (torch.norm(u_emb) + torch.norm(v_emb) + torch.norm(neg_item_emb))
                d_loss = pairwise_loss + reg_loss
                # Optimize discriminator
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
            print(self.foldInfo, 'training:', epoch + 1, 'finished.')
        print('normal training with social relations...')
        for epoch in range(1):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                u_i = np.random.randint(0, self.num_users)
                # Convert numpy arrays to PyTorch tensors
                user_idx_tensor = torch.LongTensor(user_idx)
                i_idx_tensor = torch.LongTensor(i_idx)
                j_idx_tensor = torch.LongTensor(j_idx)
                # Generate alternative neighborhood
                alternative_neighborhood = self.generator(A_tensor, u_i)
                # Forward pass through discriminator with social relations
                user_emb, item_emb = self.discriminator(norm_adj, alternative_neighborhood, True, self.attentiveTraining, self.K)
                # Get embeddings for specific users and items
                u_emb = user_emb[user_idx_tensor]
                v_emb = item_emb[i_idx_tensor]
                neg_item_emb = item_emb[j_idx_tensor]
                # Calculate loss
                y_ui = torch.sum(u_emb * v_emb, dim=1)
                y_uj = torch.sum(u_emb * neg_item_emb, dim=1)
                pairwise_loss = -torch.sum(torch.log(torch.sigmoid(y_ui - y_uj) + 1e-10))
                reg_loss = self.regU * (torch.norm(u_emb) + torch.norm(v_emb) + torch.norm(neg_item_emb))
                d_loss = pairwise_loss + reg_loss
                # Optimize discriminator
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
            print(self.foldInfo, 'training finished.')
        print('adversarial training with social relations...')
        for epoch in range(1):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                u_i = np.random.randint(0, self.num_users)
                # Convert numpy arrays to PyTorch tensors
                user_idx_tensor = torch.LongTensor(user_idx)
                i_idx_tensor = torch.LongTensor(i_idx)
                j_idx_tensor = torch.LongTensor(j_idx)
                # Generate alternative neighborhood
                alternative_neighborhood = self.generator(A_tensor, u_i)
                # Discriminator step (with adversarial loss)
                user_emb, item_emb = self.discriminator(norm_adj, alternative_neighborhood, True, self.attentiveTraining, self.K)
                u_emb = user_emb[user_idx_tensor]
                v_emb = item_emb[i_idx_tensor]
                neg_item_emb = item_emb[j_idx_tensor]
                y_ui = torch.sum(u_emb * v_emb, dim=1)
                y_uj = torch.sum(u_emb * neg_item_emb, dim=1)
                # Get current neighbors and calculate friend embeddings
                current_neighbors = alternative_neighborhood[user_idx_tensor]
                friend_embeddings = torch.mm(current_neighbors, user_emb) / self.K
                y_vi = torch.sum(friend_embeddings * v_emb, dim=1)
                pairwise_loss = -torch.sum(torch.log(torch.sigmoid(y_ui - y_uj) + 1e-10))
                reg_loss = self.regU * (torch.norm(u_emb) + torch.norm(v_emb) + torch.norm(neg_item_emb))
                adversarial_loss = -torch.sum(torch.log(torch.sigmoid(y_ui - y_vi) + 1e-10))
                d_adv_loss = pairwise_loss + reg_loss + self.beta * adversarial_loss
                self.d_optimizer.zero_grad()
                d_adv_loss.backward(retain_graph=True)
                self.d_optimizer.step()
                # Generator step
                g_adv_loss = -torch.sum(torch.log(torch.sigmoid(y_vi - y_ui) + 1e-10))
                g_loss = self.beta * g_adv_loss
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
            print(self.foldInfo, 'training:', 1//3 + 1//3 + epoch + 1, 'finished.')
            # Evaluate model after each epoch
            u_i = np.random.randint(0, self.num_users)
            alternative_neighborhood = self.generator(A_tensor, u_i)
            self.user_embeddings_final, self.item_embeddings_final = self.discriminator(norm_adj, alternative_neighborhood, False, 0, self.K)
            # Convert to numpy for evaluation
            self.U = self.user_embeddings_final.detach().cpu().numpy()
            self.V = self.item_embeddings_final.detach().cpu().numpy()
            # Evaluate performance using the parent class's ranking_performance method
            current_performance = self.ranking_performance(epoch + 2 * 1 // 3)
            # Track best embeddings - use metrics from parent class's tracking mechanism
            if self.bestPerformance and len(self.bestPerformance) > 1:
                performance_dict = {}
                for m in current_performance[1:]:
                    m = m.strip()
                    if ':' in m:
                        parts = m.split(':', 1)
                        if len(parts) == 2:
                            k, v = parts
                            try:
                                performance_dict[k.strip()] = float(v.strip())
                            except ValueError:
                                print(f"[Warning] Cannot convert value to float: '{v}' in line '{m}'")
                        else:
                            print(f"[Warning] Invalid format: '{m}'")
                    else:
                        print(f"[Warning] Skipped invalid line: '{m}'")
                current_performance_sum = sum(performance_dict.values())
                best_performance_sum = sum(self.bestPerformance[1].values())
                if current_performance_sum > best_performance_sum:
                    self.bestU = self.U.copy()
                    self.bestV = self.V.copy()
                    # saveModel() is called in the parent class's ranking_performance method
        # Use best embeddings for inference
        if self.bestU is not None and self.bestV is not None:
            self.U, self.V = self.bestU, self.bestV
        return self.ranking_performance(epoch)

    def saveModel(self):
        """Save the best performing model parameters"""
        self.bestU = self.U.copy() if self.U is not None else None
        self.bestV = self.V.copy() if self.V is not None else None

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            if self.V is None or self.U is None:
                return [self.data.globalMean] * self.num_items
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items



def load_data(path):
    return [[*line.strip().split()[:2], 1.0] for line in open(path) if line.strip()] if os.path.exists(path) else []


class ESRFTuner:
    def __init__(self, train_set, test_set, social_data, base_config):
        self.train_set = train_set
        self.test_set = test_set
        self.social_data = social_data
        self.base = base_config
        self.results = []
        self.grid = {
            'emb_size': [16, 32, 64, 128, 256, 512],
            'batch_size': [64, 128, 256, 512, 1024, 2048, 4096],
            'factors': [10, 20, 30, 40, 50],
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            'reg_lambda': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2],
            'reg_lambda_u': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2],
            'reg_lambda_i': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2],
            'reg_lambda_s': [0.001, 0.01, 0.1, 0.2, 0.5],
            'reg_lambda_b': [0.001, 0.01, 0.1, 0.2, 0.5],
            'K': [10, 20, 30],
            'beta': [0.1, 0.2, 0.3],
            'n_layer': [1, 2, 3, 4]
        }
        self.default = {
            'emb_size': 64,
            'batch_size': 256,
            'factors': 50,
            'lr': 0.001,
            'reg_lambda': 0.001,
            'reg_lambda_u': 0.001,
            'reg_lambda_i': 0.01,
            'reg_lambda_s': 0.2,
            'reg_lambda_b': 0.2,
            'K': 30,
            'beta': 0.2,
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
                    model = ESRF(conf, self.train_set, self.test_set, self.social_data)
                    metrics = model.trainModel()
                    self.results.append({'config': conf, 'metrics': metrics})
                except Exception as e:
                    print(f"[Error] Tuning {key} = {val}: {e}")
                    traceback.print_exc()
        with open('esrf_tuning_individual.json', 'w') as f:
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
            'reg_lambda_s': params['reg_lambda_s'],
            'reg_lambda_b': params['reg_lambda_b'],
            'K': params['K'],
            'beta': params['beta'],
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
        'model': {'name': 'ESRF', 'type': 'graph'},
        'output': './results/'
    }
    train_set = load_data(base_config['training.set'])
    test_set = load_data(base_config['test.set'])
    social_data = load_data(base_config['social.set'])
    print(f"Loaded {len(train_set)} training interactions")
    print(f"Loaded {len(test_set)} test interactions")
    print(f"Loaded {len(social_data)} social interactions")
    print("\nESRF Hyperparameter Tuning Framework\n" + "="*80)
    tuner = ESRFTuner(train_set, test_set, social_data, base_config)
    tuner.run()