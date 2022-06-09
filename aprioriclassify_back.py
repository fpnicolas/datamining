# 关联规则算法apriori算法
# 1生成单个元素的项集列表C1（候选项集）
# 2扫描数据集将不满足最小支持度的项集去掉，满足的构成L1
# 3对L1的元素进行组合生成两个元素的项集列表C2
# 4重复第2步
# 5依次类推
# 算法原理：如果某个项集是非频繁集，那么它的所有超集（S2是S1的子集，则S1是S2的超集）都是非频繁的
import numpy as np
import common


# 将字符串数据预处理为数字 比如beer用1表示,
def converttonumber(dataset):
    temp = []
    dict = {}
    for data in dataset:
        for i in data:
            temp.append(i)
    temp = set(temp)
    for i, r in enumerate(list(temp)):
        dict[r] = i
    for data in dataset:
        for i in range(len(data)):
            data[i] = dict[data[i]]
    return dataset, dict


# test converttonumber
# strarray = [['beer', 'paper'], ['tissue', 'gun'], ['gun', 'paper']]
# print(converttonumber(strarray))


# 将字典的key值和value值互换
def getfeats(dict):
    feats = {}
    for k, v in dict.items():
        feats[v] = k
    return feats


# 预处理过程
# 对数据分类和排序
def preprocessdata(dataset):
    dataset, dict = converttonumber(dataset)
    # todo
    for data in dataset:
        data.sort()
    dict = getfeats(dict)
    return dataset, dict


# d = preprocessdata([[2, 1], [4, 3]])
# print(d)


# 获得支持度
def getsupport(dataset, items):
    total = len(dataset)
    count = 0
    for data in dataset:
        flag = True
        for item in items:
            if item not in data:
                flag = False
                break
        if flag:
            count += 1
    return count / total


# 获得置信度 需要修改
def getconfidence(dataset, itemA, B):      # itemA是排除B的数组，B是待检测元素
    totalcount = 0
    count = 0
    for data in dataset:
        flag = True
        for A in itemA:
            if A not in data:
                flag = False
                break
        if flag:
            totalcount += 1
            if B in data:
                count += 1
    return count / totalcount


def getC1(dataset):
    result = []
    temp = []
    for data in dataset:
        for i in data:
            temp.append(i)
    temp = set(temp)
    for r in temp:
        result.append([r])
    return result


def getL1(dataset, C1, support):
    result = []
    result_No = []
    for c in C1:
        s = getsupport(dataset, c)
        if s >= support:
            result.append(c+[s])
        else:
            result_No.append(c)
    return result, result_No


def remove_thesame(arry):
    if len(arry) == 0:
        return []
    result = [arry[0]]
    for ar in arry:
        flag = True
        for r in result:
            if ar == r:
                flag = False
                break
        if flag:
            result.append(ar)
    return result


# 1,3+1,4ok, 1,3+3,5ok, 1,3+2,3no
# 我想到了两种方法，一种是生成全部，之后再剪枝，第二种是直接按一定规则生成，经过研究应该采用第一种，因此需要保留非频繁集
def getC2(L1):
    result = []
    length = len(L1[0]) + 1             # 生成的长度
    # same_num = len(L1[0]) - 1           # 相同元素的个数
    for i in range(len(L1)-1):
        j = 1
        while i + j < len(L1):
            tmp = L1[i] + L1[i+j]
            j += 1
            if len(set(tmp)) == length:
                result.append(list(set(tmp)))
    result = remove_thesame(result)
    return result


def trimC2(C2, L1_no):
    result = []
    for c in C2:
        flag = True
        for l in L1_no:
            if len(c) >= len(set(c+l)):
                flag = False
                break
        if flag:
            result.append(c)
    return result


def getapriori(dataset, minsup=0.5):      # 数据集，支持度，置信度
    result = []
    C1 = getC1(dataset)
    L1, L1_no = getL1(dataset, C1, minsup)
    while len(L1) != 0:
        # 生成C2
        C2 = getC2(L1)
        C2_trimed = trimC2(C2, L1_no)
        L1, L1_no = getL1(dataset, C2_trimed, minsup)
        result += L1
    return result


# todo 找出关联规则 这里要用到置信度了
def getrules(dataset, freqiterms, miniconfidence=0.66):
    rules = []
    confidence_group = []
    for item in freqiterms:
        for i in range(len(item)):
            itemA = item[0:i] + item[i+1:]
            B = item[i]
            confidence = getconfidence(dataset, itemA, B)
            if confidence >= miniconfidence:
                rules.append([itemA] + [B])
                confidence_group.append(confidence)
    return rules


# 用关联规则去分类
# 输入：数据， 输出一个规则，什么样的规则呢 属性值和具体标签的关联
# discreatflags = [] 0-不需要改动 1-需要离散化 2-需要将字符串转化为数字
class CBAclassifer:
    def __init__(self, dataset, feats, discreatflags, minsup=0.1, minconf=0.5):
        self.dataset = dataset
        self.feats = feats
        self.feats.append('class')                      # 加入标签的标志
        self.flags = discreatflags
        self.border = []
        self.minsup = minsup
        self.minconf = minconf
        self.y = [data[-1] for data in dataset]         # 标签数组
        self.y_len = len(set(self.y))                   # 种类数
        self.ynames = list(set(self.y))
        self.rules = []

    def preprocess(self):
        # 离散化 数字化
        for index, flag in enumerate(self.flags):
            if flag == 1:
                typename = [i for i in range(1, self.y_len+1)]
                self.border.append(common.discrete(self.dataset, index, typename))
            elif flag == 2:
                for i, data in enumerate(self.dataset):
                    for j, classname in enumerate(self.ynames):
                        if data[-1] == classname:
                            self.dataset[i][-1] = j+1
                            break
        for index, data in enumerate(self.dataset):
            for i, d in enumerate(data):
                self.dataset[index][i] = self.feats[i] + str(d)

    # 三次修剪排序规则之一 删掉不含有class的项
    def initialprune(self):
        i = 0
        while i < len(self.rules):
            if len(self.rules[i][-2]) <= 5 or self.rules[i][-2][:5] != "class":
                del (self.rules[i])
            else:
                i += 1

    def addsupport(self):
        # 加入支持率
        for i in range(len(self.rules)):
            items = self.rules[i][0].append(self.rules[i][1])
            support = getsupport(self.dataset, items)
            self.rules[i].append(support)

    # 突然想到了一个好办法，把support和confidence但列一个数组，这样便于之后的排序
    # 不是一个好办法，还是把他们放在一起，在最后生成规则后再算一遍support率
    def sortprune(self):
        pass

    def learn(self):
        # 预处理
        self.preprocess()
        # 找到频繁项
        ap = getapriori(self.dataset, self.minsup)
        self.rules = getrules(self.dataset, ap, self.minconf)
        # 修剪1 删除结果不是class的关联规则
        self.initialprune()
        print(self.rules)
        self.addsupport()
        print(self.rules)
        # 修剪2 按置信度和支持度排序
        # 修剪3 根据错误率安排规则和默认规则
        # 删掉一些rules


    def predict(self, test):
        pass


dataset, test = common.read_train_test("./iris.data", percent=0.7, randomorder=True)
feats = ['a', 'b', 'c', 'd']
cba = CBAclassifer(dataset, feats, [1, 1, 1, 1, 2])
# print(cba.dataset)
cba.learn()
# print(cba.dataset)


# 两种思路 1是直接利用现有的方法去预测
# 2 是分别为每种属性，定义不同的数字，不管怎么样，标签一定得分别定义，而且一定出现在关联规则之中。两种方法，一种是前面规定。一种是后面消减。
# 最终选择这种规则：<[A:1, B:2], class1>
# 关联规则如何生成规则呢 这是问题的关键
# 5.1,3.5,1.4,0.2,Iris-setosa
# 4.9,3.0,1.4,0.2,Iris-setosa
# 4.7,3.2,1.3,0.2,Iris-setosa
# 4.6,3.1,1.5,0.2,Iris-setosa
# 5.0,3.6,1.4,0.2,Iris-setosa
# 1离散化，记录边界，以后好进行同样的离散化
# 2引入feats 将属性和标签都做好
# 3建立关联规则
# 4筛选关联规则 删掉没有class的，删掉
# 5对数据进行预测，给出准确率


