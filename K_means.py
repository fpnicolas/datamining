# 输入无标签的数据，已经k值
# 输出多个类，也就是多个list

# 由于是无标签的，因此不需要训练集和测试集

import random
import common
import matplotlib.pyplot as plt


def del_label(dataset):
    for data in dataset:
        del(data[-1])


def float_dataset(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            dataset[i][j] = float(dataset[i][j])


def preprosse(dataset):
    k, borders = analyze_data(dataset)
    del_label(dataset)
    float_dataset(dataset)
    return k, borders

# 算法实现
# 1选择初始化的k个样本作为初始聚类中心 ；
# 2针对数据集中每个样本，计算它到k个聚类中心的距离并将其分到距离最小的聚类中心所对应的类中；
# 3针对每个类别，重新计算它的聚类中心（即属于该类的所有样本的质心）；
# 4重复上面 2 3 两步操作，直到达到某个中止条件（迭代次数、最小误差变化等）。

# 选取k个初始化数据的函数
def get_random_initial(k, datalen):
    return random.sample(range(0, datalen), k)


# 计算两个数据的欧式距离
def get_distance(x, center):
    dis = 0
    for i in range(len(x)):
        dis += (x[i] - center[i]) ** 2
    return dis


# 找到聚类中心 是最大值和最小值的中值，还是所有点之和除以个数，我准备写两个函数
def get_quality_center(dataset, flag, k):
    centers = []
    size = len(dataset[0])
    sumarray = [[0]*size for _ in range(k)]
    count = [0] * k
    for i, data in enumerate(dataset):
        for j in range(size):
            sumarray[flag[i]][j] += data[j]
        count[flag[i]] += 1
    for i, xi in enumerate(sumarray):
        center = []
        for j in xi:
            center.append(j/count[i])
        centers.append(center)
    return centers


def get_center(classarray):
    size = len(classarray[0])
    center = []
    for i in range(size):
        col = [data[i] for data in classarray]
        maxvalue = max(col)
        minvalue = min(col)
        center.append((maxvalue+minvalue)/2)
    return center


def isDistance_enough(centers, previous_centers, minupdate):
    enough = True
    for i in range(len(previous_centers)):
        dis = get_distance(centers[i], previous_centers[i])
        print(dis)
        if dis > minupdate:
            enough = False
            break
    return enough


# a = [[1, 2, 3], [3, 3, 4], [4, 5, 7], [5, 3, 2]]
# flags = [0, 1, 1, 0]
# center2 = get_quality_center(a, flags, 2)                   # 应该是[[3, 2.5, 2.5], [3.5, 4, 5.5]
# print(center2)


def drawtwopictures(dataset, flags, centers):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    colors = ['r', 'g', 'b']
    for i, flag in enumerate(flags):
        ax1.scatter(dataset[i][0], dataset[i][1], c=colors[flag])
        ax2.scatter(dataset[i][2], dataset[i][3], c=colors[flag])
    for center in centers:
        ax1.scatter(center[0], center[1], c='y')
        ax2.scatter(center[2], center[3], c='y')
    plt.show()


# 返回k的数组，还是给贴上标签呢？贴上标签的方式空间复杂度降低，因此返回一个labels数组
def get_kmeans(dataset, k, maxiteration=100, minupdate=1e-4):
    num = len(dataset)
    size = len(dataset[0])
    flags = [0] * num
    centers = []                                        # 用于记录中心点坐标，同上
    centers_index = get_random_initial(k, num)
    for index in centers_index:
        centers.append(dataset[index])
    for it in range(maxiteration):                      # 返回条件，跳出条件之一
        # if (it % 2) == 0:
        #     drawtwopictures(dataset, flags, centers)
        print(centers)
        for index, data in enumerate(dataset):          # 找到每一点的最小中心点
            mindistance = get_distance(data, centers[0])
            minindex = 0
            for index_k in range(k-1):                  # 测量与k个点的最小距离
                dis = get_distance(data, centers[index_k+1])
                if dis < mindistance:
                    mindistance = dis
                    minindex = index_k+1
            flags[index] = minindex
        preivous_centers = centers[:]                   # 用于记录上一次的中心点，是一个k维向量
        centers = get_quality_center(dataset, flags, k)
        # 退出条件之二
        if isDistance_enough(centers, preivous_centers, minupdate):
            print("足够小退出")
            # drawtwopictures(dataset, flags, centers)
            break
    return flags


# todo 如果确定三个基本点呢
# todo 中心点移动距离的可视化，干脆放在同一张图上算了。


# 输入数据，输出一个数组，记录边界，一个k值。
def analyze_data(dataset):
    k_value = 0
    borders = []
    current = None
    for i, data in enumerate(dataset):
        if data[-1] != current:
            current = data[-1]
            borders.append(i)
            k_value += 1
    return k_value, borders


# 找到各个预测范围的各种值的个数
def findmajority(flags, k):
    count = [0] * k
    for flag in flags:
        count[flag] += 1
    return count


def get_order(count):
    count_order = []
    length_count = len(count)
    for j in range(length_count):
        max = 0
        max_one = 0
        for i, c in enumerate(count):
            if count[i] > max:
                max = count[i]
                max_one = i
        count_order.append(max_one)
        count[max_one] = 0
    return count_order


def getlabels(flags, borders):
    label_k = []
    k = len(borders)
    borders.append(len(flags))        # 加入总长度以便于计算
    # 依次获得顺序值, 获得最大的那个，它的序号如果不在label_k中就加入，否则就找次大的那个，以此类推。
    # 如何找次大的比较麻烦 可以对其先排序吧，然后依次往下进行
    for i in range(k):
        count = findmajority(flags[borders[i]:borders[i+1]], k)
        count_order = get_order(count)
        for j in count_order:
            if j not in label_k:
                label_k.append(j)
                break
    # 顺序值结合border算出整个标签
    labels = []
    for i in range(len(borders)-1):
        labels += (borders[i+1] - borders[i]) * [label_k[i]]
    print(labels)
    return labels


# 评价正确率
def judge(flags, borders):
    right_count = 0
    labels = getlabels(flags, borders)
    for i in range(len(labels)):
        if flags[i] == labels[i]:
            right_count += 1
    return right_count / len(flags)


class Kmeans:
    def __init__(self, dataset, k):
        # k, borders = preprosse(dataset)
        flags = get_kmeans(dataset, k)
        print(flags)
        # print(judge(flags, borders))

if __name__ == "__main__":
    train, _ = common.read_train_test('./iris.data', percent=1)
    k, borders = analyze_data(train)
    print(k, borders)
    del_label(train)
    float_dataset(train)
    # labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    flags = get_kmeans(train, 3, minupdate=0)
    print(flags)

    print(judge(flags, borders))

