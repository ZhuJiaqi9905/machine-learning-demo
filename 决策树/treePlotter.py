import matplotlib.pyplot as plt 
#绘制树的结点
#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8") 
leafNode = dict(boxstyle="round4", fc="0.8") 
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''绘制带箭头的注解'''
    # annotate是关于一个数据点的文本  
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点  
    #annotate的作用是添加注释，nodetxt是注释的内容，
    #nodetype指的是输入的节点（边框）的形状
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, \
        textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt,parentPt,txtString):
    #作用是计算tree的中间位置    
    # cntrpt起始位置, parentpt终止位置, txtstring：文本标签信息
    #找到x和y的中间位置
    xmid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0] 
    ymid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]  
    createPlot.ax1.text(xmid,ymid,txtString)
    
    
def plotTree(myTree,parentPt,nodetxt):
    numleafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firststr=list(myTree.keys())[0]
    cntrpt=(plotTree.xoff+(1.0+float(numleafs))/2.0/plotTree.totalw,plotTree.yoff)#计算子节点的坐标 
    plotMidText(cntrpt,parentPt,nodetxt) #绘制线上的文字  
    plotNode(firststr,cntrpt,parentPt,decisionNode)#绘制节点  
    seconddict=myTree[firststr]
    plotTree.yoff=plotTree.yoff-1.0/plotTree.totald#每绘制一次图，将y的坐标减少1.0/plotTree.totald，间接保证y坐标上深度的
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            plotTree(seconddict[key],cntrpt,str(key))
        else:
            plotTree.xoff=plotTree.xoff+1.0/plotTree.totalw
            plotNode(seconddict[key],(plotTree.xoff,plotTree.yoff),cntrpt, leafNode)
            plotMidText((plotTree.xoff,plotTree.yoff),cntrpt,str(key))
    plotTree.yoff=plotTree.yoff+1.0/plotTree.totald
 
    
def createPlot(intree):
     # 类似于Matlab的figure，定义一个画布(暂且这么称呼吧)，背景为白色 
    fig=plt.figure(1,facecolor='white')
    fig.clf()    # 把画布清空 
    axprops=dict(xticks=[],yticks=[])   
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图 
    # frameon表示是否绘制坐标轴矩形 
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops) 
    
    plotTree.totalw=float(getNumLeafs(intree))
    plotTree.totald=float(getTreeDepth(intree))
    plotTree.xoff=-0.6/plotTree.totalw;plotTree.yoff=1.2;
    plotTree(intree,(0.5,1.0),'')
    plt.show()

def getNumLeafs(myTree):
    '''得到树的叶节点数目'''
    numLeafs = 0 
    firstStr = list(myTree.keys())[0] #由于dict_keys是不可索引的对象，所以转换成list 
    secondDict = myTree[firstStr] 
    for key in secondDict.keys():
        #如果结点数据类型是字典
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key]) 
        else: numLeafs += 1 
    return numLeafs 

def getTreeDepth(myTree):
    '''得到树的层数'''
    maxDepth = 0
    firstStr = list(myTree.keys())[0] 
    secondDict = myTree[firstStr] 
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = getTreeDepth(secondDict[key]) + 1
        else:
            thisDepth = 1
        maxDepth = max([maxDepth, thisDepth]) 
    return maxDepth


tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

createPlot(tree)