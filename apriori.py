# 关联规则算法apriori算法
# 1生成单个元素的项集列表C1（候选项集）
# 2扫描数据集将不满足最小支持度的项集去掉，满足的构成L1
# 3对L1的元素进行组合生成两个元素的项集列表C2
# 4重复第2步
# 5依次类推
# 算法原理：如果某个项集是非频繁集，那么它的所有超集（S2是S1的子集，则S1是S2的超集）都是非频繁的
import numpy as np


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
            result.append(c)
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


def getapriori(dataset, support=0.5):      # 数据集，支持度，置信度
    result = []
    C1 = getC1(dataset)
    L1, L1_no = getL1(dataset, C1, support)
    while len(L1) != 0:
        # 生成C2
        C2 = getC2(L1)
        C2_trimed = trimC2(C2, L1_no)
        L1, L1_no = getL1(dataset, C2_trimed, support)
        result += L1
    return result


# todo 找出关联规则 这里要用到置信度了
def getrules(dataset, freqiterms, feats, miniconfidence=0.66):
    dict_rules = {}
    for item in freqiterms:
        for i in range(len(item)):
            itemA = item[0:i] + item[i+1:]
            B = item[i]
            confidence = getconfidence(dataset, itemA, B)
            if confidence >= miniconfidence:
                itemA = [feats[a] for a in itemA]
                dict_rules['{}->{}'.format(itemA, feats[B])] = confidence
    return dict_rules


class apriror:
    def __init__(self, dataset, featdict, support=0.5, miniconfidence=0.66):
        # 预处理一下
        self.feats = featdict
        r = getapriori(dataset, support)
        self.rules = getrules(dataset, r, featdict, miniconfidence)
        print(self.rules)


if __name__ == "__main__":
    dataset = [['beer', 'nuts', 'flower'], ['log', 'nuts', 'water'], ['beer', 'log', 'nuts', 'water'], ['log', 'water']]
    # dataset = np.array(dataset, dtype=object)
    # <stdin>:1: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences \
    # (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprec \
    # ated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    dateset, feats = preprocessdata(dataset)
    r = getapriori(dataset)
    print(getrules(dataset, r, feats))

