from math import log
import matplotlib.pyplot as plt
from common import *


class C4_5:
    def __init__(self, dataset, feats):
        self.dataset = dataset
        self.feats = feats
        # 直接根据数据，建立决策树
        self.inputTree = self.createTree(self.dataset, self.feats[:])

    # 计算数据的熵
    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)                       # 数据数
        labelCounts = {}                                # 类别字典（类别的名称为键， 该类别的个数为值）
        for featVec in dataSet:
            currentLabel = featVec[-1]                  # -1是标签
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1              # 这段for代码是为了找出标签都有哪些类别，以及分别有多少个
        shannonEnt = 0.
        for key in labelCounts:                         # 求出每种类型的熵
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    # 按照指定的特征划分数据集    同时删掉了这个元素
    def splitDataSet(self, dataSet, axis, value):                       # 数据，列，值
        retDataset = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataset.append(reducedFeatVec)
        return retDataset  # 返回分类后的新矩阵

    def chooseBestFeatureToSplit(self, dataset):
        numFeatures = len(dataset[0]) - 1                               # 获得属性数目 不算标签列
        baseEntropy = self.calcShannonEnt(dataset)                      # 数据整体的熵 减号前面的部分 因此叫基础熵
        bestInfoGain = 0.
        bestFeature = -1
        for i in range(numFeatures):                                    # 求所有属性的信息增益
            featList = [example[i] for example in dataset]
            uniqueVals = set(featList)                                  # set取集合，相同的去掉
            newEntropy = 0.
            splitInfo = 0.
            for value in uniqueVals:
                subDataset = self.splitDataSet(dataset, i, value)       # 获得指定属性为某一特征值的数据
                prob = len(subDataset) / float(len(dataset))            # 上述数据的占比
                newEntropy += prob * self.calcShannonEnt(subDataset)    # 算出上述数据的熵
                splitInfo -= prob * log(prob, 2)                        # 指定属性各个特征值的熵之和
            if splitInfo == 0:                                          # 防止被除数为0
                continue
            infoGain = (baseEntropy - newEntropy) / splitInfo           # c4.5算增益率
            # print(infoGain)
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    # 找出出现次数最多的分类名称 用于作为递归的其中一种返回条件。
    def majorityCnt(self, classlist):
        sortedClassCount = []
        classCount = {}
        for vote in classlist:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=lambda cl: cl[1], reverse=True)       # sorted返回全新list
        return sortedClassCount[0][0]

    # 创建树 建立规则的过程，使用字典的嵌套实现树，这相当于学习的过程
    def createTree(self, dataset, labels):
        # 递归的返回条件 1是只有一种类型，2是没有更多属性
        classList = [item[-1] for item in dataset]
        if classList.count(classList[0]) == len(classList):  # list.count(list[0]) 计算有多少list[0]的值
            return classList[0]
        if len(dataset[0]) == 1:
            return self.majorityCnt(classList)   # 只有类别数据没有属性，返回最多类别

        bestFeat = self.chooseBestFeatureToSplit(dataset)
        bestFeatlabel = labels[bestFeat]
        myTree = {bestFeatlabel: {}}        # 通过字典的嵌套实现树
        del (labels[bestFeat])
        featValues = [e[bestFeat] for e in dataset]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]           # 删掉后的label 可以理解为深拷贝
            myTree[bestFeatlabel][value] = self.createTree(self.splitDataSet(dataset, bestFeat, value), subLabels)        # 递归
        return myTree

    # 使用决策树进行分类 这相当于使用的过程
    def classify(self, inputTree, featLabels, testVec):  # 决策树模型，标签，待分类项
        classlabel = None
        keys = list(inputTree.keys())
        firststr = keys[0]
        seconddict = inputTree[firststr]
        featIndex = featLabels.index(firststr)
        for key in seconddict.keys():
            if testVec[featIndex] == key:
                if type(seconddict[key]).__name__ == 'dict':  # 还有子树
                    classlabel = self.classify(seconddict[key], featLabels, testVec)
                else:
                    classlabel = seconddict[key]
        return classlabel

    def judge(self, dataset):
        total = len(dataset)
        right = 0
        for data_c in dataset:
            data = data_c[:-1]
            classlabel = self.classify(self.inputTree, self.feats[:], data)
            if classlabel == data_c[-1]:
                right += 1
        # print("correct percent is {}".format(right / total))
        return right / total


if __name__ == "__main__":
    # train, test = read_train_test("./car.data", percent=0.2)
    feats = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    # tree = createTree(train, feats[:])
    # print(tree)
    # judge(tree, train, feats)
    x = [0.6, 0.7, 0.8, 0.9, 1.0]
    y = []
    for i in x:
        train, test = read_train_test("./car.data", percent=i, randomorder=True)
        if len(test) == 0:
            test = train
        c4_5 = C4_5(train, feats)
        print(c4_5.inputTree)
        result = c4_5.judge(test)
        y.append(result)
    plt.scatter(x, y)
    plt.show()


# todo printTree()
# todo discrete_better()

