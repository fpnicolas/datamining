# 输入：file 应用方式 输出：结果
import c4_5, apriori, K_means
import common

filename = input("请输入文件名： ")
method = input("请输入数据挖掘方式，1 分类 2 聚类 3 关联规则: ")
method = int(method)

# 1读入文件
# 2根据不同需要method预处理数据
# 3构建对象，输出结果


if method == 1:
    percent = input("请输入训练集百分比（例子：70）： ")
    train, test = common.read_train_test(filename, percent=percent)
    featsfile = input("请输入标签名文件： ")
    feats = common.readdata(featsfile)
    c45 = c4_5.C4_5(train, feats)
    print(c45.judge(test))
elif method == 2:
    dataset = common.read_train_test(filename)
    k = input("请输入k值： ")
    K_means.Kmeans(dataset, k)
elif method == 3:
    dataset = common.read_train_test(filename)
    dataset, feats = apriori.preprocessdata(dataset)
    support = input("请输入支持率（例子：0.5）： ")
    apr = apriori.apriror(dataset, feats, support=support)
