# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:13:33 2017
预处理获取小波系数、重构系数、模极大值
对预处理后的序列进行特征提取，包括能量、归一化能量、能量熵、能量矩、排列熵、近似熵、样本熵、最大Lyapunov指数、灰度矩
分形维数、奇异熵
@author: baishuhua
"""

import pywt
import numpy as np
from scipy import stats
import sys
sys.path.append('E:\\大数据\\基础研究\\HilbertHuang变换')
import HilbertHuang
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
        
def DWT(signal,wavelet,level): 
    # 实数序列进行dwt变换，返回的是不等长的小波系数，[cAn,cDn,cDn-1,...,cD2,cD1]
#    max_level=pywt.dwt_max_level(data_len=len(signal), filter_len=pywt.Wavelet(wavelet).dec_len)
    Coeffs=pywt.wavedec(signal,wavelet=wavelet,mode='smooth',level=level)
    return Coeffs
    
def DwtRec(signal,wavelet,level): 
    # 返回的是每一层的分量，即将原始信号进行了频带分解，时间长度与原始序列signal等长
    # 每一层对应一行
    CoeffsRev=[]
    Coeffs=DWT(signal,wavelet=wavelet,level=level)
    for row in range(level+1):
        if row==0:
            # 重构近似系数，即低频分量
            leveltype=level;part='a'
        else:
            # 重构细节系数，即高频分量
            leveltype=level-row+1;part='d'
        EachLevelRev=pywt.upcoef(part,Coeffs[row],wavelet=wavelet,level=leveltype,take=len(signal))
        CoeffsRev.append(EachLevelRev)
    return np.array(CoeffsRev)

def DwtModMax(signal,wavelet,level):
    # 返回的是细节系数的模极大值序列
    DetailCoeffs=DWT(signal,wavelet=wavelet,level=level)[1:]
    ExtremeX=[];ExtremeY=[]
    for coeff in DetailCoeffs:
        pos=HilbertHuang.findpeaks(coeff)
        neg=HilbertHuang.findpeaks(-coeff)
        xx=np.concatenate((pos,neg))
        yy=np.concatenate((coeff[pos],coeff[neg]))
        index=np.argsort(xx)
        ExtremeX.append(list(xx[index]));ExtremeY.append(list(yy[index]))
    return np.array(ExtremeX),np.array(ExtremeY)
    
def GetEnergy(coeffs):
    # 能量'Energy'
    Energy=[];EnergyName=[]
    for order,EachLevelCoeff in enumerate(coeffs):
        Energy.append(np.sum(EachLevelCoeff**2))
        EnergyName.append('Energy-L'+str(len(coeffs)-order))
    return np.array(EnergyName),np.array(Energy)

def GetEnergyRatio(coeffs):
    # 归一化能量'EnergyRatio'
    EnergyName,Energy=GetEnergy(coeffs)
    EnergyRatioName=list(map(lambda x:x.replace('Energy','EnergyRatio'),EnergyName))
    EnergyRatio=Energy/np.sum(Energy)
    return np.array(EnergyRatioName),np.array(EnergyRatio)

def Entropy(pk): # 香农熵,pk参数非负代表的是概率
    entropy=stats.entropy(pk,qk=None,base=None)
    return entropy

def GetEnergyEn(coeffs):
    # 能量熵'EnergyEn'
    WEE=[];WEEName=[]
    for order,coeff in enumerate(coeffs):
        pk=coeff**2/np.sum(coeff**2)
        WEE.append(Entropy(pk))
        WEEName.append('EnergyEn-L'+str(len(coeffs)-order))
    return np.array(WEEName),np.array(WEE)

def GetEnergyTq(coeffs):
    # 归一化能量矩'EnergyTq'
    WET=[];WETName=[]
    for order,coeff in enumerate(coeffs):
        weight=np.linspace(1,len(coeff),num=len(coeff))
        WET.append(np.dot(weight,coeff**2))
        WETName.append('EnergyTq-L'+str(len(coeffs)-order))
    WET/=np.sum(WET)
    return np.array(WETName),np.array(WET)

def GetGrayTq(coeffs):
    # 灰度矩'GrayTq'
    m,n=np.shape(coeffs)
    M=0
    for i,j in zip(np.arange(1,m+1),np.arange(1,n+1)):
        M+=abs(coeffs[i-1,j-1])*np.sqrt((i-1)**2+(j-1)**2)
    M/=(m*n)
    return np.array(['GrayTq']),np.array([M])

def GetSingularEn(coeffs):
    # 奇异熵'SingularEn'
    U,s,V=np.linalg.svd(coeffs,full_matrices=True)
    pk=s/np.sum(s)
    WSE=Entropy(pk)
    return np.array(['SingularEn']),np.array([WSE])
  
def GetFractalDims(coeffs):
    # 分形维数'FractalDims'
    Ref0=np.log2(np.var(np.sum(coeffs,axis=0))) # 原信号方差的对数
    temp=coeffs[1:]
    xx=np.arange(len(coeffs)-1,0,step=-1).ravel()
    yy=np.log2(np.var(temp,axis=1)).ravel()
    yy=np.insert(yy,0,Ref0);xx=np.insert(xx,0,0)
    index=np.argsort(xx)
    xx=xx[index];yy=yy[index]
    params=np.polyfit(xx,yy,1)
    beta=-params[0];D=1.0/2*(5-beta)
    return np.array(['FractalDims']),np.array([D])  

def EuclideanDistance(x1,x2):
    return np.sqrt(np.dot(x1-x2,x1-x2))

def DistanceMatrix(X,method=EuclideanDistance):
    Distance=np.zeros((len(X),len(X)))
    for row in np.arange(0,len(X)-1):
        Xi=X[row]
        for col in np.arange(row,len(X)):
            Xj=X[col]
            Distance[row,col]=method(Xi,Xj)
    Distance+=np.transpose(Distance)
    return Distance
    
def SearchNearest(Distance):
    # 按照行，返回与之距离最近的行编号和距离值
    XPos0=[];YDist0=[]
    for row,dist in enumerate(Distance):
        temp=np.argsort(dist)
        index=temp[temp!=row][0]
        XPos0.append(index)
        YDist0.append(dist[index])
    return XPos0,YDist0
   
def GetLyapunov(coeffs):
    # 最大Lyapunov指数 'Lyapunov'
    if len(coeffs)==np.size(coeffs):
        coeffs=coeffs[np.newaxis,:]

    m=3;J=5 # 分别对应嵌入维数和延迟参数
    N=np.shape(coeffs)[-1] # 对应时间序列的长度
    M=N-(m-1)*J # 对应相空间重构矩阵的行数
    Lamda=[];LamdaName=[]
    for order,coeff in enumerate(coeffs):
        X=[] # 对应每层小波系数重构的相空间
        for row in range(m):
            X.append(coeff[int(row*J):int(row*J+M)])
        X=np.transpose(np.array(X))
        
        Distance=DistanceMatrix(X)
        XPos0,YDist0=SearchNearest(Distance) 
        # 第i行代表与第i个样本距离最近的样本索引及其最近距离，公式中的j^与dj(0)
        XRef0=np.arange(0,len(Distance))
        # 与XPos0对应的参考样本索引,公式中的j
        lamda=[]
        for indexRef0,indexPos0 in zip(XRef0,XPos0):
            xx=[0];yy=[np.log(YDist0[indexRef0]+1e-10)]
            for bias in range(min(len(XRef0)-indexRef0,len(XRef0)-indexPos0)-1):
                xx.append(bias+1)
                yy.append(np.log(Distance[indexRef0+bias+1,indexPos0+bias+1]+1e-10))
            
            if len(xx)>1:
                params=np.polyfit(xx,yy,1)
                lamda.append(params[0])
            else:
                continue
        Lamda.append(max(lamda))
        LamdaName.append('Lyapunov-L'+str(len(coeffs)-order))
    return np.array(LamdaName),np.array(Lamda)  

def AbsDistance(signal1,signal2):
    return np.max(np.abs((signal1-signal2)))

def GetApEn(coeffs):    
    if len(coeffs)==np.size(coeffs):
        coeffs=coeffs[np.newaxis,:]
    ApEn=[]; # 对应近似熵
    
    PHY=[];
    N=np.shape(coeffs)[-1] # 对应时间序列的长度
    for m in [2,3]: # 对应嵌入维数
        M=N-m+1 # 对应重构矩阵的行数
        PHY_before=[];ApEnName=[]
        for order,coeff in enumerate(coeffs):
            gama=0.2*np.std(coeff) # 对应相似容限
            X=[] # 对应每层小波系数重构的相空间
            for row in range(M):
                X.append(coeff[int(row):int(row+m)])
            Distance=DistanceMatrix(X,method=AbsDistance)
            
            total=list(map(lambda x:sum(x<=gama)-1,Distance))
            Ci=np.divide(total,float(M-1))
            Ci=Ci[np.nonzero(Ci)]            
            PHY_before.append(np.mean(np.log(Ci))) # 对应phy-m
            ApEnName.append('ApEn-L'+str(len(coeffs)-order))
        PHY.append(PHY_before)
    PHY=np.array(PHY)
    ApEn=PHY[0]-PHY[1] 
    
    return np.array(ApEnName),np.array(ApEn)

def GetSampEn(coeffs):    
    if len(coeffs)==np.size(coeffs):
        coeffs=coeffs[np.newaxis,:]
    SampEn=[] # 对应样本熵
    
    B=[]
    N=np.shape(coeffs)[-1] # 对应时间序列的长度
    for m in [2,3]: # 对应嵌入维数
        M=N-m+1 # 对应重构矩阵的行数
        B_before=[];SampEnName=[]
        for order,coeff in enumerate(coeffs):
            gama=0.2*np.std(coeff) # 对应相似容限
            X=[] # 对应每层小波系数重构的相空间
            for row in range(M):
                X.append(coeff[int(row):int(row+m)])
            Distance=DistanceMatrix(X,method=AbsDistance)
            
            total=list(map(lambda x:sum(x<=gama)-1,Distance))
            Ci=np.divide(total,float(M-1))
            Ci=Ci[np.nonzero(Ci)]            
            B_before.append(np.mean(Ci))
            SampEnName.append('SampEn-L'+str(len(coeffs)-order))
        B.append(B_before)
    B=np.array(B)
    SampEn=-np.log(B[1]/B[0]) # 分别对应近似熵和样本熵
    return np.array(SampEnName),np.array(SampEn)

def GetPermuEn(coeffs):
    # 排列熵 'PermutationEn'
    if len(coeffs)==np.size(coeffs):
        coeffs=coeffs[np.newaxis,:]

    m=3;J=5 # 分别对应嵌入维数和延迟参数
    N=np.shape(coeffs)[-1] # 对应时间序列的长度
    M=N-(m-1)*J # 对应相空间重构矩阵的行数
    PermuEn=[];PermuEnName=[]
    for order,coeff in enumerate(coeffs):
        X=[] # 对应每层小波系数重构的相空间
        for row in range(m):
            X.append(coeff[int(row*J):int(row*J+M)])
        X=np.transpose(np.array(X))
        S=list(map(lambda x:np.argsort(x),X))
        S=np.array(S)
        
        strS=list(map(str,S))
        strCode=list(map(lambda x:'$'.join(x),strS))
        
        Counts={}
        for each in strCode:
            if each not in Counts.keys():
                Counts[each]=1
            else:
                Counts[each]+=1
        pk=np.divide(list(Counts.values()),len(S))
        PermuEn.append(Entropy(pk)) 
        PermuEnName.append('PermuEn-L'+str(len(coeffs)-order))           
    return np.array(PermuEnName),np.array(PermuEn)
        
# 综合Dwt变换域指标

def FeatureExtractDwt(signal):
#    FunctionNames=[GetEnergy,GetEnergyRatio,GetEnergyEn,GetEnergyTq,GetGrayTq,GetSingularEn,\
#                   GetFractalDims,GetLyapunov,GetApEn,GetSampEn,GetPermuEn]
    # 仅提取部分特征
    FunctionNames=[GetSingularEn,GetGrayTq,GetFractalDims] 
#    if np.size(signal)==len(signal):
#        signal=signal[:,np.newaxis]
    ResultName=[];Result=[]
    for functionname in FunctionNames:
        featurename,feature=functionname(signal)
        ResultName.append(featurename)
        Result.append(feature)
    FeatureValue=np.concatenate(Result)
    FeatureName=np.concatenate(ResultName)
    return FeatureName,FeatureValue
      
# 示例

def ReadRecord(file): # 读取录波数据
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
    FaultIndex=int(ReadFlag.readline().strip().split('fault_index:')[1])
    FaultTime=Time[FaultIndex]
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
    return Time,Data,np.array(SignalNames),FaultTime

def Interp(x,y,kind='slinear',axis=0): # 沿axis轴拟合x，y，因变量y沿axis维度应等于x维度
    from scipy import interpolate
    function=interpolate.interp1d(x,y,kind,axis)
    return function

if __name__=='__main__':
    # 模拟数据
    if 0:
#        fs=1000
#        time=np.arange(0,1,step=1/fs)
#        signal=2*np.cos(2*np.pi*50*time)+4+0.25*np.cos(2*np.pi*100*time)+1*np.cos(2*np.pi*150*time)
        fs=1000;f0=50;f1=350;f2=150
        time=np.arange(0,10,step=1/fs);signal=6*np.cos(2*np.pi*f0*time)+2+0.25*np.cos(2*np.pi*f2*time)
        t1=4;t2=6;signal[(time>=t1)&(time<=t2)]+=0.05*np.cos(2*np.pi*f1*time[(time>=t1)&(time<=t2)])
    
    # 仿真数据
    if 1:
        file = r'E:\关于项目\江苏涌流识别\仿真\bsh\内部故障\5%匝_AB_[0.1 0.12]s.txt'
        ReadFlag = open(file)
        data = []
        for line in ReadFlag:
            attr = line.strip().split(' ')
            attr = list(map(float,attr))
            data.append(attr)
        data = np.array(data)
        signal = data[:,1]
        time = data[:,0]
        fs = 1/(time[1]-time[0])
        
        coeffs = DwtRec(signal,wavelet='bior3.5',level=4)
        FeatureName,FeatureValue =  FeatureExtractDwt(coeffs) 
        
    # 实测数据
    if 0:
        import os
        file='E:/大数据/线路故障诊断/解压录波/有效集中128/Test-1/异物_1_0_12_港城站_20140510港遂乙线_SSH850.cfg_sub.dat_AN'
        filename=os.path.basename(file)
        attr=filename.split('_')
        FaultCause=attr[0];FaultPhaseNum=attr[1];FaultPhaseType=attr[-1]
        Time,Data,SignalNames,FaultTime=ReadRecord(file)
        
        SelectRow=(Time>=FaultTime-0.06)&(Time<=FaultTime+0.1) # 截取故障前2个周波，故障后10个周波进行分析
        Time=Time[SelectRow];Data=Data[SelectRow]
        SelectCol=np.array([False,False,False,True,False,False,False,False])
        Data=Data[:,SelectCol] # 选择A相电流
        fitting=Interp(Time,Data,kind='slinear',axis=0)
    
        fs=1000 # 插值固定的采样率        
        TimeNew=np.arange(min(Time),max(Time),step=1.0/fs)[:-1]
        DataNew=fitting(TimeNew)
        time=TimeNew;signal=DataNew.ravel()
    
        
        
        
    if 1:
        
        wavelet='bior3.5';level=4   
        coeffsRev=DwtRec(signal,wavelet,level)
        fig,axes=plt.subplots(level+1,1,figsize=(6,6))
        for row,ax in enumerate(axes.ravel(),start=1):
            ax.plot(time,coeffsRev[level+1-row]);ax.set_ylabel('尺度'+str(row))
#        axes[0].set_title(FaultCause)
        plt.show()
    
        if 1:
            plt.plot(time,signal);plt.ylabel('原始信号');
            yrange=max(signal)-min(signal)
            plt.ylim([min(signal)-1/10*yrange,max(signal)+1/10*yrange])
            plt.show()


     