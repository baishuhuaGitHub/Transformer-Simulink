# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:59:30 2018
不均衡样本集重采样——imbalance-learn
@author: baishuhua
"""

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, \
AllKNN, RepeatedEditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour, OneSidedSelection, \
NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold

from imblearn.combine import SMOTETomek, SMOTEENN

from imblearn.ensemble import EasyEnsemble, BalanceCascade

#from imblearn.ensemble import BalancedBaggingClassifier # 组合采样和分类器

from collections import Counter

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
# import shutil

# 列写全部样本列表及对应的标签
def AllList(Path, Suffix):
    Sets = ['close', 'open', 'middle', 'virual']
    All_X, All_Y = [], []
    for index, classname in enumerate(Sets):
        lists = glob.glob(os.path.join(Path, classname, Suffix))
        All_X.extend(lists)
        All_Y.extend([index]*len(lists))
    All_X, All_Y = np.array(All_X), np.array(All_Y)
    return All_X, All_Y

# 拆分训练集和测试集
def TrainTestSplit(X, Y, test_size=0.2, random_state=42):
    #random_state = np.random.seed()
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=test_size, \
                                                        random_state=random_state)
    return train_X, test_X, train_Y, test_Y

# 过采样-重抽样训练集
'''
    对于数值型样本集，可用方法有：'Random'、'SMOTE'、'SMOTE_border1'、'SMOTE_border2'、
    'SMOTE_svm'、'ADASYN'
    
    对于非数值型样本集，可用方法有：'Random'
'''
def OverSample(X, Y, method='Random', random_state=42):
    if X.size == len(X):
        X = X.reshape(-1,1)
    if method is 'Random':
        sampler = RandomOverSampler(ratio='auto', random_state=random_state)
    elif method is 'SMOTE':
        sampler = SMOTE(ratio='auto', random_state=random_state)
    elif method is 'SMOTE_border1':
        sampler = SMOTE(ratio='auto', random_state=random_state, kind='borderline1')
    elif method is 'SMOTE_border2':
        sampler = SMOTE(ratio='auto', random_state=random_state, kind='borderline2')
    elif method is 'SMOTE_svm':
        sampler = SMOTE(ratio='auto', random_state=random_state, kind='svm')
    elif method is 'ADASYN':
        sampler = ADASYN(ratio='auto', random_state=random_state)
    X_resampled, Y_resampled = sampler.fit_sample(X, Y)
    return X_resampled, Y_resampled

# 欠采样-重采样训练集
'''
    对于数值型样本集，可用方法有：'Cluster';'Random'、'NearMiss_1'、'NearMiss_2'、
    'NearMiss_3'（欠采样后样本数定）;'TomekLinks'、'ENN'、'RENN'、'All_KNN'（欠采样后样本数不定）；
    'CNN'、'One_SS'、'NCR'；'IHT'（训练模型后删除预测概率低的样本）
    
    对于非数值型样本集，可用方法有：'Random'
'''

def UnderSample(X, Y, method='Random', random_state=42):
    if X.size == len(X):
        X = X.reshape(-1,1)
    if method is 'Cluster': # 默认kmeans估计器
        sampler = ClusterCentroids(ratio='auto', random_state=random_state, estimator=None)
    elif method is 'Random':
        sampler = RandomUnderSampler(ratio='auto', random_state=random_state, replacement=False)
    elif method is 'NearMiss_1':
        sampler = NearMiss(ratio='auto', random_state=random_state, version=1)
    elif method is 'NearMiss_2':
        sampler = NearMiss(ratio='auto', random_state=random_state, version=2)
    elif method is 'NearMiss_3':
        sampler = NearMiss(ratio='auto', random_state=random_state, version=3)
    elif method is 'TomekLinks':
        sampler = TomekLinks(ratio='auto', random_state=random_state)
    elif method is 'ENN': # kind_sel可取'all'和'mode'
        sampler = EditedNearestNeighbours(ratio='auto', random_state=random_state, kind_sel='all')
    elif method is 'RENN': # kind_sel可取'all'和'mode'
        sampler = RepeatedEditedNearestNeighbours(ratio='auto', random_state=random_state, kind_sel='all')
    elif method is 'All_KNN':
        sampler = AllKNN(ratio='auto', random_state=random_state, kind_sel='all')
    elif method is 'CNN':
        sampler = CondensedNearestNeighbour(ratio='auto', random_state=random_state)
    elif method is 'One_SS':
        sampler = OneSidedSelection(ratio='auto', random_state=random_state)
    elif method is 'NCR':
        sampler = NeighbourhoodCleaningRule(ratio='auto', random_state=random_state, kind_sel='all', threshold_cleaning=0.5)
    elif method is 'IHT':
        sampler = InstanceHardnessThreshold(estimator=None, ratio='auto', random_state=random_state)
    X_resampled, Y_resampled = sampler.fit_sample(X, Y)
    return X_resampled, Y_resampled

# 组合过/欠采样
'''
    对于数值型样本集，可用方法有：'SMOTETomek'、'SMOTEENN'
    
    对于非数值型样本集，无可用方法
'''

def OverAndUnderSample(X, Y, method='SMOTEENN', random_state=42):
    if X.size == len(X):
        X = X.reshape(-1,1)
    if method is 'SMOTETomek': 
        sampler = SMOTETomek(ratio='auto', random_state=random_state, smote=None, tomek=None, k=None, m=None, out_step=None, kind_smote=None)
    elif method is 'SMOTEENN':
        sampler = SMOTEENN(ratio='auto', random_state=random_state, smote=None, enn=None, k=None, m=None, out_step=None, kind_smote=None, kind_enn=None)
    X_resampled, Y_resampled = sampler.fit_sample(X, Y)
    return X_resampled, Y_resampled

# 组合采样和分类器
'''
    对于数值型样本集，可用方法有：'EasyEnsemble'、'BalanceCascade'
    
    对于非数值型样本集，可用方法有：'EasyEnsemble'
'''

def EnsembleSample(X, Y, method='EasyEnsemble', random_state=42):
    if X.size == len(X):
        X = X.reshape(-1,1)
    if method is 'EasyEnsemble':
        sampler = EasyEnsemble(ratio='auto', random_state=random_state, replacement=False, n_subsets=10)
    elif method is 'BalanceCascade':
        sampler = BalanceCascade(ratio='auto', random_state=random_state, n_max_subset=None, classifier=None, estimator=None)
    X_resampled, Y_resampled = sampler.fit_sample(X, Y)
    # 组合采样+分类器，返回的是分类器
#    BalancedBaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, ratio='auto', replacement=False, n_jobs=1, random_state=None, verbose=0)
    return X_resampled, Y_resampled

# 测试不同的重采样技术
if __name__ == '__main__':
    DataPath = r'E:\大数据\深度学习\刀闸状态识别\刀闸状态标记'
    DataSuffix = '*.jpg'
    Lists, Labels = AllList(DataPath, DataSuffix)
    print("type of Lists is %s, shape of Lists is %s" %(type(Lists), Lists.shape,))

    Train_X, Test_X, Train_Y, Test_Y = TrainTestSplit(Lists, Labels)
    print("shape of trainset is %s" %(Train_Y.shape,))
    print("shape of testset is %s" %(Test_Y.shape,))

    print('details of trainset is {}'.format(sorted(Counter(Train_Y).items())))

    X_oversampled, Y_oversampled = OverSample(Train_X, Train_Y)
    print('details of oversampled trainset is {}'.format(sorted(Counter(Y_oversampled).items())))
    
    X_undersampled, Y_undersampled = UnderSample(Train_X, Train_Y)
    print('details of undersampled trainset is {}'.format(sorted(Counter(Y_undersampled).items())))
    
    if 0:
        X_combsampled, Y_combsampled = OverAndUnderSample(Train_X, Train_Y)
        print('details of undersampled trainset is {}'.format(sorted(Counter(Y_combsampled).items())))
    
    X_ensembsampled, Y_ensembsampled = EnsembleSample(Train_X, Train_Y)
    print('details of undersampled trainset is {}'.format(sorted(Counter(Y_ensembsampled[0]).items())))