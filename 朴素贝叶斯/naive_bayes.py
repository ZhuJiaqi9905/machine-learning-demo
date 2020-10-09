#coding=utf-8
import numpy as np 
import os
import random
import math
import time


def readData(rootDir):
    '''将文本文件读入到dataSet中
    参数：rootDir 文件存放的目录
    返回：dataSet 存放词条的数据集
          labels 每个样本的标签
    '''
    dataSet = []
    labels = []
    #folderList: 根目录下的所有子目录，如“财经”，“房产”等
    folderList = os.listdir(rootDir)
    #遍历每个子目录
    for folder in folderList:
        dataDir = os.path.join(rootDir, folder)
        files = os.listdir(dataDir) 
        #打开子目录下的文本文件
        for file in files:
            txtDir = os.path.join(dataDir, file)  #每一个文本文件的地址
            #读入数据
            with open(txtDir, "r", encoding="utf-8", errors="ignore") as f:
                data = list(f.read().split())
                dataSet.append(data)
            labels.append(folder) 
    return dataSet, labels

def readStopWords(stopWordsDir):
    ''' 读取停词表
    参数：stopWordsDir 存放停词表文件的目录
    返回：stopWords 停词表
    '''
    with open(stopWordsDir, "r", encoding="utf-8", errors="ignore") as f:
        stopWords = list(f.read().split())
    return stopWords

def splitDataSet(dataSet, labels, numFolds):
    '''把数据集和标签划分成numFolds份，本实验是十折交叉验证，所以numFolds=10
    输入：dataSet: 数据集
          labels：数据集中每个样本的标签
          numFolds 分割成的份数，本次是10份
    输出：
        等分的数据集和对应的标签
    '''
    #把数据集和标签打乱，要保证打乱的顺序是一样的
    state = np.random.get_state()
    np.random.shuffle(dataSet) 
    np.random.set_state(state)
    np.random.shuffle(labels) 
    #返回分割成numFolds份的列表
    return np.array_split(dataSet, numFolds), np.array_split(labels, numFolds)
        

def preprocessText(dataSet, stopWords, featNum):
    '''文本预处理：将训练集中所有词条按出现频率从大到小排序。并从中选择词条，组成词汇表
    输入：trainSet 训练集
         stopWords 停词表
         featNum 选取的词条数目
    输出：finalWordList 选取的词汇表
    '''
    num = 0
    wordDict = {}
    #统计每个词条的出现次数
    for sample in dataSet:
        for word in sample:
            wordDict[word] = wordDict.get(word, 0) + 1
    #按出现次数从大到小排序
    preWordList = sorted(wordDict.items(), key = lambda item: item[1], reverse=True)
    preWordList = list(preWordList)
    preWordList, _ = zip(*preWordList)
    #选取featNum个词条，组成最终的词汇表
    finalWordList = []
    num = 0
    for word in preWordList:
        if  (word not in stopWords) and len(word) > 1:
            finalWordList.append(word)
            num += 1
        if num >= featNum:
            break
    return finalWordList


def setWord2Vec(dataSet, wordList, dataLabels, className):
    '''
    根据词汇表，把数据集中的词条转化为向量
    输入：dataSet 数据集
          wordList 词汇表
          dataLabels 文本形式的标签
          className ['财经', '房产', '健康', '教育', '军事', '科技', '体育', '娱乐', '证券']
    输出：dataVec 由数据集构成的词向量集
        labelVec 对应于数字的标签
    '''
    length = len(wordList)
    dataVec = []
    #遍历数据集中每个样本
    for sample in dataSet:
        vec = [0] * length
        for word in sample:
            if word in wordList:
                #如果样本出现这条词汇，就在向量对应位置+1
                vec[wordList.index(word)] += 1
        dataVec.append(vec)
    labelsLength = len(dataLabels)
    labelVec = np.zeros((labelsLength), dtype=np.int32)
    #遍历每个标签，把文本标签对应成向量
    for i in range(labelsLength):
        labelVec[i] = className.index(dataLabels[i])
    return np.array(dataVec), np.array(labelVec)

def trainModel(trainSet, trainLabels, className):
    '''利用训练集计算朴素贝叶斯中的概率，从而训练模型
    输入：trainSet 训练集，是词向量的形式
        trainLebels 训练集每条样本的标签
        className ['财经', '房产', '健康', '教育', '军事', '科技', '体育', '娱乐', '证券']
    输出：pClass, pClass[i]表示第i类文档在总文档中出现的概率
         pWord, #pWord[i][j]表示词汇表的第j个词，在第i类文档中出现的概率
         由于概率值相乘容易造成数值过小，所以这里取了对数
    '''
    sampleNum = len(trainSet) #训练集中的样本数目
    featNum = len(trainSet[0]) #特征数目，就是每个词向量的长度
    classNum = len(className) #分类类别的个数，本实验中共9类
    wordSum = np.array(np.ones((classNum, featNum))) #wordSum[i][j]表示第i类文档中，在词汇表中第j个词出现的次数
    sampleInClass = np.array(np.ones((classNum)))  #记录每一类文档在训练集中出现的次数
    wordsInClass = 2 * np.array(np.ones((classNum))) #wordsInClass[i]指第i类文档中的总单词数

    #遍历每一个样本
    for i in range(sampleNum):
        classIndex = trainLabels[i] #这个样本所属类别的标号
        sampleInClass[classIndex] += 1
        wordSum[classIndex] += trainSet[i] 
        wordsInClass[classIndex] += sum(trainSet[i]) 
    #计算朴素贝叶斯模型中的概率
    pClass = np.array(np.zeros(classNum)) #pClass[i]表示第i类文档在总文档中出现的概率
    pWord = np.array(np.zeros((classNum, featNum))) #pWord[i][j]表示词汇表的第j个词，在第i类文档中出现的概率
    for i in range(classNum):
        pClass[i] = 1.0 * sampleInClass[i] / (sampleNum + classNum)
        pWord[i] =  1.0 * wordSum[i] / wordsInClass[i] 
    pClass = np.log(pClass) #对概率值取对数
    pWord = np.log(pWord) 
    return pClass, pWord



def testModel(testSet, testLabels, pClass, pWord):
    '''测试模型的准确率
    输入：testSet 测试集，是词向量的形式
         testLabels 测试集每个样本的标签
         pClass, pClass[i]表示第i类文档在总文档中出现的概率
         pWord, #pWord[i][j]表示词汇表的第j个词，在第i类文档中出现的概率
    输出：模型成功的概率
    '''
    accuracy = 0
    #计算后验概率
    prob = np.dot(testSet, pWord.transpose()) + pClass
    #统计预测正确的数目
    accuracy = np.sum(np.argmax(prob, axis=1) == testLabels)
    testNum = len(testSet)
    return 1.0 * accuracy / testNum 


#数据集和停词表的路径。用相对路径可能会找不到，最好用绝对路径
dataDir = "D:\\Study\\2020-03\\3_人工智能引论\\实践课\\朴素贝叶斯\\new_weibo_13638\\" 
stopWordsDir = "D:\\Study\\2020-03\\3_人工智能引论\\实践课\\朴素贝叶斯\\stop_words.txt"


#读取训练数据和停词表
dataSet, labels = readData(dataDir)
stopWords = readStopWords(stopWordsDir)
className = ['财经', '房产', '健康', '教育', '军事', '科技', '体育', '娱乐', '证券']

#文本预处理，提取特征
featNum = 4000    #选取的特征数量
wordList = preprocessText(dataSet, stopWords, featNum) 

#把文本数据转换为向量的形式
dataVec, dataLabels = setWord2Vec(dataSet, wordList, labels, className)

#十折交叉验证
numFolds = 10
XFolds, yFolds = splitDataSet(dataVec, dataLabels, numFolds)#XFolds和yFolds存的是把数据集分成的10等份

accuracies = np.zeros((numFolds))

#进行十折交叉验证
for i in range(numFolds):
    #把数据集的第i份作为测试集，其余作为训练集
    trainSet = np.vstack(XFolds[0: i] + XFolds[i+1 : ])
    trainLabels = np.hstack(yFolds[0: i] + yFolds[i+1 : ])
    trainLabels = trainLabels.reshape((-1))
    testSet = XFolds[i]
    testLabels = yFolds[i].reshape((-1)) 
    #训练模型，得到先验概率和条件概率
    pClass, pWord = trainModel(trainSet, trainLabels, className)
    #测试模型，输出准确率
    accuracies[i] = testModel(testSet, testLabels, pClass, pWord)
    print("第 %d 次，准确率为 %.2f%%" % (i+1, 100*accuracies[i]))

print("交叉验证的平均准确率为 %.2f%%" %(100 * np.sum(accuracies)/numFolds))





