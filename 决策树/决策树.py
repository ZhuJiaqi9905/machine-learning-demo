from math import log 
import operator
import pickle 
def calShannonEnt(dataSet):
    '''计算集合的香侬熵'''
    numEntries = len(dataSet)
    labelCounts = {} #记录每种标签的个数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1 
        else:
            labelCounts[currentLabel] += 1 
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt += -prob * log(prob, 2) 
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''按照给定特征划分数据集
    axis 特征, 
    value 特征的取值'''
    
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #把数据集中axis的特征抽走
            reducedFeatVec = featVec[ : axis]
            reducedFeatVec.extend(featVec[axis+1 :]) #注意这是extend
            #样本加入新的集合
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''选取最好的数据集划分方式'''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #找出特征i的所有值
        featList = [example[i] for example in dataSet] 
        uniqueVals = set(featList)
        newEntropy = 0.0 
        for value in uniqueVals:
            #划分数据集
            subDataSet = splitDataSet(dataSet, i, value)
            #计算划分后引起的熵变化
            prob = len(subDataSet) / float(len(dataSet)) 
            newEntropy += prob * calShannonEnt(subDataSet) 
        #计算信息增益
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i 
    return bestFeature 

def majorityCnt(classList):
    '''
    计算某一特征中出现频率最高的取值,
    classList 样本中某个特征的取值列表
    '''
    classCount = {} 
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1 
    #按出现频率从高到低排序
    #classCount.items()返回一个可遍历的(键, 值) 元组数组
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    '''构造决策树，返回值是类别标签'''
    classList = [example[-1] for example in dataSet] 
    #如果数据集中全是相同标签的样本，则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0] 
    #如果使用完所有特征，仍不能划分完全，就返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList) 
    #找到当前最好的划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #树的结构是一个多层嵌套的字典 
    myTree = {bestFeatLabel:{}} 
    del(labels[bestFeat])
    #得到该属性的所有属性值
    featValues = [example[bestFeat] for example in dataSet] 
    uniqueVals = set(featValues) 
    for value in uniqueVals:
        subLabels = labels[:] #复制标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree 

def classify(inputTree, featLabels, testVec):
    '''分类器'''
    firstStr = list(inputTree.keys())[0] 
    secondDict = inputTree[firstStr] 
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec) 
            else:
                classLabel = secondDict[key] 
    return classLabel 

def storeTree(inputTree, filename):
    fw = open(filename, 'wb') 
    pickle.dump(inputTree, fw) 
    fw.close()
    
def grabTree(filename):
    fr = open(filename, 'rb')
    tree = pickle.load(fr)  
    fr.close() 
    return tree 

def creatDateSet():
    dataSet = [ [1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers'] 
    return dataSet, labels 

dataSet , labels = creatDateSet()

myTree = createTree(dataSet, labels[:]) 
storeTree(myTree, "./test.txt") 
#newTree = grabTree("./test.txt") 
#print( classify(newTree, labels, [1, 1]) ) 
#print(myTree)