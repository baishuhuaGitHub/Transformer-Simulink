# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:11:55 2019

@author: baishuhua
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
import os
import glob
from scipy import interpolate
import DwtFeatureExtraction
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import time

import sys;sys.path.append('E:\\大数据') 
import sampler 
from collections import Counter

'''
# ********** 第一阶段：原始数据展示及预处理 ********** 
'''

# 读文本数据,返回Time，Signals，其中Signals按列依次代表一次侧三相电流和二次侧三相电流
def ReadData_Text(file):
    Time = []
    Signals = []
    with open(file) as f:
        for line in f:
            Attr = line.strip().split(' ')
            Content = list(map(float, Attr))
            Time.append(Content[0])
            Signals.append(Content[1:])
        return np.array(Time), np.array(Signals)    

def WaveForm_Cut(t, y, before=0.04, after=0.2):
    y_abs = np.abs(y)
    x_peak = np.argmax(y_abs, axis=0)
    left,right = min(x_peak[:3]),max(x_peak[:3])
    interval = np.logical_and(t>=(t[left]-before), t<=(t[right]+after))
    return t[interval], y[interval,:]

# 展示原始波形数据   
def ViewWaveform_TimeDomain(t, y, issave=True, name=''):
    fig,axes = plt.subplots(figsize=(10,6), nrows=2, ncols=2, sharex=True)
    axes[0,0].plot(t, y[:,:3]);axes[0,0].legend(['Ia','Ib','Ic']);axes[0,0].set_ylabel('一次侧')
    axes[0,1].plot(t, y[:,:3]-y[:,[1,2,0]]);axes[0,1].legend(['Iab','Ibc','Ica'])
    axes[1,0].plot(t, y[:,3:]);axes[1,0].legend(['Ia','Ib','Ic']);axes[1,0].set_ylabel('二次侧')
    axes[1,1].plot(t, y[:,3:]-y[:,[-1,-2,-3]]);axes[1,1].legend(['Iab','Ibc','Ica'])
#    plt.xlim([0.0,0.6])
    plt.suptitle(os.path.basename(name))
    if issave:
        plt.savefig(name)
        plt.close()

# 数据预处理，插值成固定采样率的时间序列
def Interp(x, y, delta=1/1000, kind='slinear', axis=0): # delta为相邻自变量的间隔
    # 沿axis轴拟合x，y，因变量y沿axis维度应等于x维度
    function=interpolate.interp1d(x, y, kind, axis)
    x_new = np.arange(min(x), max(x)-delta, step=delta)     
    y_new = function(x_new)
    return x_new, y_new

# 现场录波读取，注意数据书写格式，前三行电压电流通道信息，后变采样率信息
def ReadFaultRecord(file): # 读取录波数据
    ReadFlag=open(file)
    ReadFlag.readline();ReadFlag.readline();ReadFlag.readline()
    segnum=ReadFlag.readline().strip().split(':')[1]
    Time=[]
    for seg in range(int(segnum)):
        SampleAttr=ReadFlag.readline().strip().split(' ')
        [fs,start,terminal]=list(map(int,map(float,SampleAttr)))
        if len(Time)<1:
            Time.extend(1/fs*(np.linspace(1,terminal-start+1,num=terminal-start+1)))
        else:
            Time.extend(Time[-1]+1.0/fs*(np.linspace(1,terminal-start+1,num=terminal-start+1)))
        
    ReadFlag.readline();ReadFlag.readline() # 跳过故障相信息
    SignalNames=ReadFlag.readline().strip().split(' ')
    SignalNames=[name for name in SignalNames if name!='']    
    Data=[];ValidIndex=[];row=0
    for record in ReadFlag:
        detail=record.strip().split(' ')
        if len(detail)==len(SignalNames):
            try:
                Data.append(list(map(float,detail)))
                ValidIndex.append(row)
                row=row+1
            except ValueError:
                row=row+1
        else:
            row=row+1
            continue
    ReadFlag.close()
    ValidIndex=np.array(ValidIndex)
    rows=ValidIndex[ValidIndex<len(Time)]    
    Time=np.array(Time);Time=Time[rows];Data=np.array(Data);Data=Data[rows];
    return Time,Data

# 动模试验仿真，注意数据书写格式，先变采样率信息，后所有模拟通道
def ReadSimuRecord(file): # 读取仿真数据
    ReadFlag=open(file)
    segnum=ReadFlag.readline().strip().split(':')[1]
    Time=[]
    for seg in range(int(segnum)):
        SampleAttr=ReadFlag.readline().strip().split(' ')
        [fs,start,terminal]=list(map(int,map(float,SampleAttr)))
        if len(Time)<1:
            Time.extend(1/fs*(np.linspace(1,terminal-start+1,num=terminal-start+1)))
        else:
            Time.extend(Time[-1]+1.0/fs*(np.linspace(1,terminal-start+1,num=terminal-start+1)))
        
    SignalNames=ReadFlag.readline().strip().split(' ')
    SignalNames=[name for name in SignalNames if name!='']   
    SignalNames=SignalNames[:41]
    Data=[];ValidIndex=[];row=0
    for record in ReadFlag:
        detail=record.strip().split(' ')
        detail=[value for value in detail if value!='']  
        if len(detail)==len(SignalNames):
            try:
                Data.append(list(map(float,detail)))
                ValidIndex.append(row)
                row=row+1
            except ValueError:
                row=row+1
        else:
            row=row+1
            continue
    ReadFlag.close()
    ValidIndex=np.array(ValidIndex)
    rows=ValidIndex[ValidIndex<len(Time)]    
    Time=np.array(Time);Time=Time[rows];Data=np.array(Data);Data=Data[rows];
    return Time,Data

'''
# ********** 第二阶段：特征工程，生成特征集和类别标签 **********
'''

# 特征提取
def ExtractFeature(file):
    time, signals = ReadData_Text(file)
    time_new, signals_new = Interp(time, signals)
    signals_input = signals_new - signals_new[:,[1,2,0,4,5,3]] # 差流信号        
    names_input = ['Iab1','Ibc1','Ica1','Iab2','Ibc2','Ica2'] # 1,2代表一，二次侧    
    featurename = [];featurevalue = []
    for col in range(np.shape(signals_new)[-1]):
        sample = signals_input[:,col] 
        
        # ********** DWT小波变换域特征提取 ********** #
        wavelet = 'bior3.5';level = 4
        coeffs = DwtFeatureExtraction.DwtRec(sample, wavelet=wavelet, level=level)
        name,value = DwtFeatureExtraction.FeatureExtractDwt(coeffs) 
        
        # 每个特征名添加对待处理物理量的描述
        name_new = list(map(lambda x:names_input[col]+'-'+x,name))
        featurename.extend(name_new)
        featurevalue.extend(value)   
    featurename=np.array(featurename);featurevalue=np.array(featurevalue)
    return featurename, featurevalue

# 特征样本集生成，并将其保存至文件中
def ExtractFeatures_AllFiles(FilePath, SavePath, filename='Features.csv'): 
    print('Start extracting features ...\n')
    BigFeatures = [];BigSamplenames = [];BigLabels = []
    SaveFile = os.path.join(SavePath, filename)
    
    import csv
    file_object = open(SaveFile, 'w', newline='')
    writer=csv.writer(file_object)
    
    FileLists = glob.glob(FilePath, recursive=True)
    FileNums=len(FileLists)
    for fileNo,file in enumerate(FileLists, start=1):
        try:
            featurename, featurevalue=ExtractFeature(file)
            filename=os.path.basename(file)
            attr=filename.split('_')[0]
        
            BigFeatures.append(featurevalue) # 特征矩阵
            BigSamplenames.append(file) # 样本名列表
            BigLabels.append(attr) # 标签列表
        except:
            continue
        
        if fileNo == 1:
            head = np.concatenate((np.array(['Filename','Label']), featurename))
            writer.writerow(head)
        line=np.concatenate((np.array([filename,attr]), featurevalue))
        writer.writerow(line)
        
        if fileNo%1000 == 0 & fileNo != FileNums:     
            print('Completed {0} %'.format(round(fileNo/FileNums*100,4)))
        elif fileNo == FileNums:
            print('Finish extracting features!!!')
            
    BigFeatures = np.array(BigFeatures)
    BigSamplenames = np.array(BigSamplenames)
    BigLabels = np.array(BigLabels)
    
    file_object.close()
    return BigSamplenames, featurename, BigFeatures, BigLabels

# 装载已经提取的特征集
def LoadDataSet(file):
    Data = pd.read_csv(file, encoding='gbk', na_values=['#NAME?','inf','INF','-inf','-INF']) 
    featurenames = Data.columns.values[2:]
    filenames = Data['Filename'].values
    labels = Data['Label'].values

    FeatureFrame = Data[featurenames]
    FeatureFrame = FeatureFrame.dropna(how='all',axis=0)
    FeatureFrame = FeatureFrame.fillna(axis='index',method='pad')
    features = FeatureFrame.values
    return filenames, featurenames, features, labels

'''
# ********** 第三阶段：Machine Learning 模型训练 **********
'''

# 对类别标签进行编码
def Encoding(Label):
    Encoder = preprocessing.LabelEncoder().fit(Label)
    return Encoder

# 特征集进行标准化
def Scaler():
    scaler = preprocessing.MinMaxScaler(copy=True)
    return scaler

'''
# ********** 第四阶段：Machine Learning 模型评估 **********
'''

# 选择评价标准
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,f1_score,fbeta_score,jaccard_similarity_score,precision_score,recall_score # 该类指标越大性能越好
from sklearn.metrics import hamming_loss,zero_one_loss  # 该类指标绝对值越小性能越好
def Evaluate(score_func=accuracy_score):
    if score_func in [accuracy_score,f1_score,fbeta_score,jaccard_similarity_score,precision_score,recall_score]:
        greater_is_better = True
    elif score_func in [hamming_loss,zero_one_loss]:
        greater_is_better = False
    score = make_scorer(score_func=score_func, greater_is_better=greater_is_better, needs_proba=False, \
                        needs_threshold=False, average='weighted')  #
    return score

# 可视化二分类器的ROC曲线，并计算曲线积分面积
from sklearn.metrics import roc_curve,auc
def Plot_ROC_binary(y_true, y_score, view):    
    '''
    y_true：真实标签，y_score：决策值，view：是否展示ROC曲线
    '''
    fpr,tpr,thresholds = roc_curve(y_true, y_score, pos_label=None)
    area=auc(fpr, tpr, reorder=False)
    if view:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC and AUC is %.5f' %area)
        plt.legend(loc='lower right')
        plt.xlim([0.0,1.0]);plt.ylim([0.0,1.1])
        plt.xlabel('假正类');plt.ylabel('真正类')
        plt.title('ROC曲线')
        plt.show()
    return area

# 可视化多分类器的ROC曲线，并加权计算曲线积分面积
def Plot_ROC_multi(ys_orig, ys_true, ys_scores, classnames, view):
    '''
    ys_orig：多类别真实标签，一列，多个类别标记，如[0,2,3,1,4]
    ys_true：对ye_orig进行二进制编码后的多列，每一列中只有一类为正，其余多类均为负
    ys_scores:与ys_true对应的多列决策值
    classnames:ROC展示时每根曲线对应的类名
    '''
    fpr = dict();tpr = dict();roc_auc = dict()
    for i,classname in enumerate(classnames):
        y_true = ys_true[:,i];y_score = ys_scores[:,i]
        if sum(y_true)==0: 
            continue
        else:
            fpr[i],tpr[i],thresholds = roc_curve(y_true, y_score, pos_label=None)
            area = auc(fpr[i], tpr[i], reorder=False)
            roc_auc[i] = area    
    
    # Compute weighted-average ROC curve and ROC area,对于weighted模式，各类的加权系数为各类在总类中的占比
    all_fpr = np.unique(np.concatenate([fpr[key] for key in fpr.keys()]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    from scipy import interp
    for key in fpr.keys():
        mean_tpr += interp(all_fpr, fpr[key], tpr[key])*(sum(ys_orig==key)/len(ys_orig))
    # Finally compute AUC
    fpr['weighted'] = all_fpr;tpr['weighted'] = mean_tpr
    roc_auc['weighted'] = auc(fpr['weighted'], tpr['weighted'])
    if view:
        plt.figure()
        for key in fpr.keys():
            try:
                classname = classnames[key]
                plt.plot(fpr[key], tpr[key], label=classname+' ROC and AUC is %.5f' %(roc_auc[key]))
            except:
                continue
        plt.plot(fpr['weighted'], tpr['weighted'], label='综合 ROC and AUC is %.5f' %(roc_auc['weighted']))
        plt.legend(loc='lower right')
        plt.xlim([0.0,1.0]);plt.ylim([0.0,1.1])
        plt.show()    
    return fpr, tpr, roc_auc['weighted']

# 可视化混淆矩阵
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.brg):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    cm:混淆矩阵,classes:混淆矩阵显示时各类的标记名称
    """
    np.set_printoptions(precision=5)    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/(cm.sum(axis=1)[:,np.newaxis]+1e-10)
        print("Normalized confusion matrix")
    else:
        print('Unnormalized confusion matrix')    
    print(cm)

    thresh=cm.max()/ 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # itertools.product笛卡尔积   代表两层（嵌套循环）
        plt.text(j, i, round(cm[i,j],5), horizontalalignment="center", color="black" if cm[i,j]>thresh else "white")
    plt.tight_layout()
    plt.ylabel('真实标签') # True label
    plt.xlabel('预测标签') # Predicted label
    
# 主程序
if __name__ == '__main__':
    FilePath = r'E:\关于项目\江苏涌流识别\仿真\ML试验\*.txt'
    SavePath = r'E:\关于项目\江苏涌流识别\仿真'
    # 是否重新进行特征提取
    if 0: 
        filenames, featurenames, features, labels = \
        ExtractFeatures_AllFiles(FilePath, SavePath, filename='Features_Inrush_Fault_Cut.csv')
    # 是否装载已构建的特征集
    if 1:
        filenames, featurenames, features, labels = \
        LoadDataSet(os.path.join(SavePath, 'Features_Inrush_Fault_Cut.csv'))
    
    # 选择待分析工况
    if 1: 
#        ignore = (labels=='恢复涌流')#|(labels=='和应涌流')
#        filenames, features, labels = \
#        filenames[~ignore], features[~ignore,:], labels[~ignore]
        select = (labels=='空投涌流')|(labels=='和应涌流')|(labels=='恢复涌流')
        labels[select]='涌流'
        select = (labels=='绕组故障匝地')|(labels=='绕组故障匝间')|(labels=='绕组故障相间')|(labels=='引线故障')
        labels[select]='内部故障'   
                 
    X = np.array(features, dtype=np.float64);X = X[:,:9]
    Encoder = Encoding(labels)
    Y = Encoder.fit_transform(labels)
    
    # 拆分训练集和测试集
    seed = 42 # np.random.seed()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)        
    print('Category distribution in TrainSet is {}'.format(sorted(Counter(Y_train).items())))
    print('*'*60,'\n')
    print('Category distribution in TestSet is {}'.format(sorted(Counter(Y_test).items())))
    print('*'*60,'\n')
    
    # 是否对训练集进行均衡化处理
    if 1:
        if 0: # 过采样
            X_oversampled, Y_oversampled = sampler.OverSample(X_train, Y_train)
            X_train, Y_train = X_oversampled, Y_oversampled
        '''
            欠采样method可选如下：
            对于数值型样本集，有：'Cluster';'Random'、'NearMiss_1'、'NearMiss_2'、
            'NearMiss_3'（欠采样后样本数定）;'TomekLinks'、'ENN'、'RENN'、'All_KNN'（欠采样后样本数不定）
            'CNN'、'One_SS'、'NCR'；'IHT'（训练模型后删除预测概率低的样本）       
            对于非数值型样本集，有：'Random'
        '''
        if 1: # 欠采样
            X_undersampled, Y_undersampled = sampler.UnderSample(X_train, Y_train, method='Random')
            X_train, Y_train = X_undersampled, Y_undersampled
        
        print('Category distribution in TrainSet After balancing is {}'.format(\
              sorted(Counter(Y_train).items())))
        print('*'*60,'\n')

    # 分类器构建    
    scaler = Scaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = 'LogisticRegression'
    score_func = f1_score # 性能评价指标 f1_score
    from sklearn.linear_model import LogisticRegression
    
    # ######################### 未进行参数优化 #########################
    if 0:
        print(' '.join(['*'*25,model,'*'*25]))
        start = time.time()
        clf = LogisticRegression(random_state=seed)
        clf = clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
#        print('Predict Results: ',Encoder.inverse_transform(Y_pred))
#        print('True Results: ',Encoder.inverse_transform(Y_test))
#        print('Score is {}'.format(clf.score(X_test,Y_test))) # 单次拆分性能指标
        judge = cross_val_score(clf, X, Y, groups=None, scoring=Evaluate(score_func), cv=5)
#        print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean())) # 交叉验证平均性能指标
        print('Total running time is {}s'.format(time.time()-start))
    
    # ######################### 网格搜索优化 #########################
    if 1:
        print(' '.join(['*'*25,model,'GridSearch','*'*25]))
        start = time.time()
        param_grid = [{'C':np.logspace(-5,2,num=8)}] 
        clf0 = GridSearchCV(estimator=LogisticRegression(random_state=seed), param_grid=param_grid, scoring=Evaluate(score_func),\
                                     fit_params=None, n_jobs=1, refit=True, cv=5, return_train_score=True)
        clf0 = clf0.fit(X_train, Y_train)
        clf = clf0.best_estimator_
#        print("Best parameters: {}".format(clf.best_params_))
#        print("Best cross-validation score: {}".format(grid_search.best_score_))
#        print('Score is {}'.format(clf.score(X_test,Y_test)))
        judge=cross_val_score(clf, X, Y, groups=None, scoring=Evaluate(score_func), cv=5)
        print('Cross-validation score is {}'.format(judge))
        print('Mean cross-validation score is {}'.format(judge.mean())) # 可替换成与交叉验证相同的评价指标，待更新
        print('Total running time is {}s'.format(time.time()-start))
            
     # 混淆矩阵评估
    if 1:
        from sklearn.metrics import confusion_matrix
        y_true = Y_test
        y_pred = clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=np.unique(Y_train))
        classes = Encoder.inverse_transform(np.unique(Y_train))
        plot_confusion_matrix(cnf_matrix, classes, normalize=False, title='混淆矩阵') 
        
    # ROC评估
    if 0: # 二分类问题
        y_true = Y_test
        y_scores = clf.decision_function(X_test)        
        area = Plot_ROC_binary(y_true, y_scores, view=1)
        
    if 0: # 多分类问题
        from sklearn.preprocessing import LabelBinarizer
        
        lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False).fit(Y)        
        classes = lb.classes_
        classnames = Encoder.inverse_transform(classes)
        
        YY = lb.transform(Y)
        XX = X
        XX_train, XX_test, YY_train, YY_test=train_test_split(XX, YY, test_size=0.25, random_state=seed)
        
        from sklearn.multiclass import OneVsRestClassifier        
        estimator = OneVsRestClassifier(clf, n_jobs=1)
        estimator = estimator.fit(XX_train, YY_train)

        ys_scores = estimator.decision_function(XX_test)
        Y_test = lb.inverse_transform(YY_test)
        # Y_test代表多分类标签列表，YY_test代表真实的二进制编码后的标签数组，ys_scores为决策输出
        # classnames为二进制编码每列对应的类别标记
        area = Plot_ROC_multi(Y_test, YY_test, ys_scores, classnames, 1)
        
                
     # 测试现场录波的预测效果
    if 1: # 现场录波
        newfiles = r'E:\关于项目\江苏涌流识别\数据\实测波形\20180610南和线和侧\*.cfg.sub'
        lists=glob.glob(newfiles)
        results=[]
        for newfile in lists[:1]:
            print('开始进行涌流和内部故障识别**********\n')
            Time, Data = ReadFaultRecord(newfile)
            Data = Data[:,3:6]
            # 截取一段时间内的数据进行分析
            
            Data = Data[Time<=min([max(Time),1.0]),:] 
            Time = Time[Time<=min([max(Time),1.0])]
            
#            Time, Data = WaveForm_Cut(Time, Data, before=0.2, after=0.4) 
            
            plt.figure();plt.plot(Time,Data);plt.title(os.path.basename(newfile))
            plt.xlabel('时间/s');plt.ylabel('幅值/kA');plt.legend(['Ia','Ib','Ic'])
            window = 0.18
            # 滑动数据窗进行涌流和内部故障识别
            begins = np.arange(min(Time), max(Time)-window, step=0.01)
            period = dict(); predict = dict(); confidence = dict()
            
            output = dict()
            for Nowindow,begin in enumerate(begins):
                DataNew = Data[(Time>=begin) & (Time<begin+window),:] 
                TimeNew = Time[(Time>=begin) & (Time<begin+window)]
            
                TimeNew, DataNew = Interp(TimeNew, DataNew)
                tnew, snew = TimeNew, DataNew[:,:3]-DataNew[:,[1,2,0]]
                
                featurenew = []
                try:
                    for col in range(np.shape(snew)[-1]):
                        sample = snew[:,col]                 
                        # ********** DWT小波变换域特征提取 ********** #
                        wavelet = 'bior3.5';level = 4
                        coeffs = DwtFeatureExtraction.DwtRec(sample, wavelet=wavelet, level=level)
                        name,value = DwtFeatureExtraction.FeatureExtractDwt(coeffs) 
                        featurenew.extend(value)   
                    featurenew = np.array(featurenew)
                    featurenew.astype(dtype = np.float64)
                                                
                    xx = featurenew.reshape(1,-1)                    
                    xx = scaler.transform(xx)                
                    result = clf.predict(xx)[0]
                    
                    period[Nowindow] = [round(min(TimeNew),2),round(max(TimeNew),2)]
                    predict[Nowindow] = Encoder.inverse_transform(result)
                    confidence[Nowindow] = clf.predict_proba(xx)[0,result]
                    
                    print('时段：',round(min(TimeNew),2),round(max(TimeNew),2),\
                          '预测类别：',Encoder.inverse_transform(result),\
                          '置信度：',clf.predict_proba(xx)[0,result])
                    
                    if Encoder.inverse_transform(result) in output.keys():
                        if confidence[Nowindow]>output[Encoder.inverse_transform(result)][1]:
                            output[Encoder.inverse_transform(result)]=(\
                              [round(min(TimeNew),2),round(max(TimeNew),2)],\
                              clf.predict_proba(xx)[0,result],featurenew)
                    else:
                        output[Encoder.inverse_transform(result)]=(\
                              [round(min(TimeNew),2),round(max(TimeNew),2)],\
                              clf.predict_proba(xx)[0,result],featurenew)
                    
                except:
#                    print('Feature too big or too small in Current Window!!!')
                    continue
            
            aa = list(period.values())
            bb = list(predict.values())
            cc = list(confidence.values())
            
            mm = [list(v)[0] for k,v in itertools.groupby(bb)]
            nn = [len(list(v)) for k,v in itertools.groupby(bb)]
        
            print(mm,nn,output)
            
                
        if 0: # 动模试验   
            newfiles = r'E:\关于项目\江苏涌流识别\数据\扬州北仿真波形\扬州北波形\*1AE2.cfg.etrall'
            lists=glob.glob(newfiles)
            results=[]
            for newfile in lists[:1]:
                print('开始进行涌流和内部故障识别**********\n')
                Time, Data = ReadSimuRecord(newfile) 
                # 选择变压器三相电流
                Data = Data[:,24:27]  # 28:31
                # 截取一段时间内的数据进行分析
                
                Data = Data[Time<=min([max(Time),1.0]),:] 
                Time = Time[Time<=min([max(Time),1.0])]
                
    #            Time, Data = WaveForm_Cut(Time, Data, before=0.2, after=0.4) 
                
                plt.figure();plt.plot(Time,Data);plt.title(os.path.basename(newfile))
                plt.xlabel('时间/s');plt.ylabel('幅值/kA');plt.legend(['Ia','Ib','Ic'])
                
                window = 0.18
                # 滑动数据窗进行涌流和内部故障识别
                begins = np.arange(min(Time), max(Time)-window, step=0.01)
                period = dict(); predict = dict(); confidence = dict()
                
                output = dict()
                for Nowindow,begin in enumerate(begins):
                    DataNew = Data[(Time>=begin) & (Time<begin+window),:] 
                    TimeNew = Time[(Time>=begin) & (Time<begin+window)]
                
                    TimeNew, DataNew = Interp(TimeNew, DataNew)
                    tnew, snew = TimeNew, DataNew[:,:3]-DataNew[:,[1,2,0]]
                    
                    featurenew = []
                    try:
                        for col in range(np.shape(snew)[-1]):
                            sample = snew[:,col]                 
                            # ********** DWT小波变换域特征提取 ********** #
                            wavelet = 'bior3.5';level = 4
                            coeffs = DwtFeatureExtraction.DwtRec(sample, wavelet=wavelet, level=level)
                            name,value = DwtFeatureExtraction.FeatureExtractDwt(coeffs) 
                            featurenew.extend(value)   
                        featurenew = np.array(featurenew)
                        featurenew.astype(dtype = np.float64)
                                                    
                        xx = featurenew.reshape(1,-1)                    
                        xx = scaler.transform(xx)                
                        result = clf.predict(xx)[0]
                        
                        period[Nowindow] = [round(min(TimeNew),2),round(max(TimeNew),2)]
                        predict[Nowindow] = Encoder.inverse_transform(result)
                        confidence[Nowindow] = clf.predict_proba(xx)[0,result]
                        
                        print('时段：',round(min(TimeNew),2),round(max(TimeNew),2),\
                              '预测类别：',Encoder.inverse_transform(result),\
                              '置信度：',clf.predict_proba(xx)[0,result])
                        
                        if Encoder.inverse_transform(result) in output.keys():
                            if confidence[Nowindow]>output[Encoder.inverse_transform(result)][1]:
                                output[Encoder.inverse_transform(result)]=(\
                                  [round(min(TimeNew),2),round(max(TimeNew),2)],\
                                  clf.predict_proba(xx)[0,result],featurenew)
                        else:
                            output[Encoder.inverse_transform(result)]=(\
                                  [round(min(TimeNew),2),round(max(TimeNew),2)],\
                                  clf.predict_proba(xx)[0,result],featurenew)
                        
                    except:
                        print('Feature too big or too small in Current Window!!!')
                        continue
                
                aa = list(period.values())
                bb = list(predict.values())
                cc = list(confidence.values())
                
                mm = [list(v)[0] for k,v in itertools.groupby(bb)]
                nn = [len(list(v)) for k,v in itertools.groupby(bb)]
            
                print(mm,nn,output)
    
    
    # 仿真算例
    if 1: 
        newfiles = r'E:\关于项目\江苏涌流识别\数据\CT饱和\浐东2线\*.txt'
        lists=glob.glob(newfiles)
        results=[]
        for newfile in lists[:1]:
            print('开始进行涌流和内部故障识别**********\n')
            Time, Data = ReadData_Text(newfile)
            Data = Data[:,3:6]
            # 截取一段时间内的数据进行分析
            partial = (Time>=0.0) & (Time<=1.65)
            Data = Data[partial,:] 
            Time = Time[partial]
            
#            Time, Data = WaveForm_Cut(Time, Data, before=0.2, after=0.4) 
            
            plt.figure();plt.plot(Time,Data);plt.title(os.path.basename(newfile))
            plt.xlabel('时间/s');plt.ylabel('幅值/kA');plt.legend(['Ia','Ib','Ic'])
            window = 0.18
            # 滑动数据窗进行涌流和内部故障识别
            begins = np.arange(min(Time), max(Time)-window, step=0.01)
            period = dict(); predict = dict(); confidence = dict()
            
            output = dict()
            for Nowindow,begin in enumerate(begins):
                DataNew = Data[(Time>=begin) & (Time<begin+window),:] 
                TimeNew = Time[(Time>=begin) & (Time<begin+window)]
            
                TimeNew, DataNew = Interp(TimeNew, DataNew)
                tnew, snew = TimeNew, DataNew[:,:3]-DataNew[:,[1,2,0]]
                
                featurenew = []
                try:
                    for col in range(np.shape(snew)[-1]):
                        sample = snew[:,col]                 
                        # ********** DWT小波变换域特征提取 ********** #
                        wavelet = 'bior3.5';level = 4
                        coeffs = DwtFeatureExtraction.DwtRec(sample, wavelet=wavelet, level=level)
                        name,value = DwtFeatureExtraction.FeatureExtractDwt(coeffs) 
                        featurenew.extend(value)   
                    featurenew = np.array(featurenew)
                    featurenew.astype(dtype = np.float64)
                                                
                    xx = featurenew.reshape(1,-1)                    
                    xx = scaler.transform(xx)                
                    result = clf.predict(xx)[0]
                    
                    period[Nowindow] = [round(min(TimeNew),2),round(max(TimeNew),2)]
                    predict[Nowindow] = Encoder.inverse_transform(result)
                    confidence[Nowindow] = clf.predict_proba(xx)[0,result]
                    
                    print('时段：',round(min(TimeNew),2),round(max(TimeNew),2),\
                          '预测类别：',Encoder.inverse_transform(result),\
                          '置信度：',clf.predict_proba(xx)[0,result])
                    
                    if Encoder.inverse_transform(result) in output.keys():
                        if confidence[Nowindow]>output[Encoder.inverse_transform(result)][1]:
                            output[Encoder.inverse_transform(result)]=(\
                              [round(min(TimeNew),2),round(max(TimeNew),2)],\
                              clf.predict_proba(xx)[0,result],featurenew)
                    else:
                        output[Encoder.inverse_transform(result)]=(\
                              [round(min(TimeNew),2),round(max(TimeNew),2)],\
                              clf.predict_proba(xx)[0,result],featurenew)
                    
                except:
#                    print('Feature too big or too small in Current Window!!!')
                    continue
            
            aa = list(period.values())
            bb = list(predict.values())
            cc = list(confidence.values())
            
            mm = [list(v)[0] for k,v in itertools.groupby(bb)]
            nn = [len(list(v)) for k,v in itertools.groupby(bb)]
        
            print(mm,nn,output)    
    
    # 随机抽取几个样本展示其三相电流及差动电流
    if 0:
        FilePath = r'E:\关于项目\江苏涌流识别\仿真\ML试验\绕组故障*.txt'
        FileLists = glob.glob(FilePath, recursive=True)
        FileLists = np.array(FileLists)
        select = np.random.randint(len(FileLists), size=100)
        NewList = FileLists[select]
#        NewList = FileLists
        for index,file in enumerate(NewList, start=1):
            try:
                Time,Signals = ReadData_Text(file)
                ViewWaveform_TimeDomain(Time, Signals, issave=True, name=file.replace('.txt','.png'))
                # 截取有效数据段
#                Time,Signals = WaveForm_Cut(Time, Signals, before=0.1, after=0.2)
#                ViewWaveform_TimeDomain(Time, Signals, issave=True, name=file.replace('.txt','.png'))
            except:
                print('Invalid file!!!')
            if index%1000 == 0 & index != len(NewList):
                print('Completed {0} %\n'.format(index/len(NewList)*100))
            elif index == len(NewList):
                print('Completed 100%\n')
    