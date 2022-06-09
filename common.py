import random


def readdata(file, sep=','):
    dataset = []
    with open(file, 'r') as f:
        lines = f.readlines()  # 只读文件一次
        for line in lines:
            words = line.split(sep=sep)
            words[-1] = words[-1].strip()
            dataset.append(words)
    return dataset


def read_train_test(file, sep=',', percent=1, randomorder=False):
    train_data = []
    test_data = []
    with open(file, 'r') as f:
        lines = f.readlines()                               # 只读文件一次
        if randomorder:
            random.shuffle(lines)
        border = int(len(lines) * percent)
        # print(border)
        for line in lines[:border]:
            words = line.split(sep=sep)
            words[-1] = words[-1].strip()
            train_data.append(words)
        for line in lines[border:]:
            words = line.split(sep=sep)
            words[-1] = words[-1].strip()
            test_data.append(words)
    return train_data, test_data


# 离散化的函数
# 根据typenames数组元素的个数将某个特征值分成相应的部分
def discrete(dataset, index, typenames):
    featlist = [item[index] for item in dataset]
    for i in range(len(featlist)):
        featlist[i] = float(featlist[i])
    maxvalue = max(featlist)
    minvalue = min(featlist)
    classnum = len(typenames)
    distance = maxvalue - minvalue              # 增量
    borders = []
    for i in range(classnum-1):                 # 分n类中间需要n-1个分界
        borders.append(minvalue + (distance/classnum*(i+1)))
    borders.append(maxvalue)                    # 增加最大值，便于下面操作
    for vect in dataset:
        for i, border in enumerate(borders):
            if float(vect[index]) <= border:
                vect[index] = typenames[i]
                break
    return borders


# 用于测试离散化函数discrete()
# testdt = [[10], [20], [40], [70], [100], [34], [56], [90]]
# typename = ['low', 'mid', 'high']
# discrete(testdt, 0, typename)
# print(testdt)

# 处理缺省值
# 对于缺失值的处理，从总体上来说分为删除存在缺失值的个案和缺失值插补
def deldefault(dataset):
    tobedel = []
    for i, data in enumerate(dataset):
        flag = False
        for item in data:
            if item == '':
                flag = True
                break
        if flag:
            tobedel.insert(0, i)
            continue
    for i in tobedel:
        del(dataset[i])


# 用于测试处理缺省值的代码
# lines = readdata("./car.data")
# deldefault(lines)
# print(lines)
