# 写在最前面的话
# kmeans 不同于其他算法，它是一种无监督的数据挖掘算法
# 其核心是去寻找最佳的聚类数，但是实际问题中最佳聚类数可能会比较大，故在此之谈算法去找这个最佳的K
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics

#构建一个3聚类的虚拟数据
np.random.seed(1234)
mean1 = [0.5,0.5]
cov1 = [[0.3,0],[0,0.3]]
x1,y1 = np.random.multivariate_normal(mean1,cov1,1000).T
mean2 = [0,8]
cov2 = [[1.5,0],[0,1]]
x2,y2 = np.random.multivariate_normal(mean2,cov2,1000).T
mean3 = [8,4]
cov3 = [[1.5,0],[0,1]]
x3,y3 = np.random.multivariate_normal(mean3,cov3,1000).T
data = pd.DataFrame(np.concatenate([np.array([x1,y1]),np.array([x2,y2]),np.array([x3,y3])],axis=1).T)

# 自定义寻找最佳k
# 拐点法
def k_SSE(X, clusters):
    # 选择连续的K种不同的值
    K = range(1,clusters+1)
    # 构建空列表用于存储总的簇内离差平方和
    TSSE = []
    for k in K:
        # 用于存储各个簇内离差平方和
        SSE = []
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # 返回簇标签
        labels = kmeans.labels_
        # 返回簇中心
        centers = kmeans.cluster_centers_
        # 计算各簇样本的离差平方和，并保存到列表中
        for label in set(labels):
            SSE.append(np.sum((X.loc[labels == label,]-centers[label,:])**2))
        # 计算总的簇内离差平方和
        TSSE.append(np.sum(SSE))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    plt.plot(K, TSSE, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('簇内离差平方和之和')
    plt.show()
# 轮廓系数法
def k_silhouette(X, clusters):
    K = range(2, clusters + 1)
    # 构建空列表，用于存储个中簇数下的轮廓系数
    S = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        # 调用字模块metrics中的silhouette_score函数，计算轮廓系数
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制K的个数与轮廓系数的关系
    plt.plot(K, S, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('轮廓系数')
    # 显示图形
    plt.show()
# 间隔统计量法
def short_pair_wise_D(each_cluster):
    mu = each_cluster.mean(axis=0)
    Dk = sum(sum((each_cluster - mu) ** 2)) * 2.0 * each_cluster.shape[0]
    return Dk
def compute_Wk(data, classfication_result):
    Wk = 0
    label_set = set(classfication_result)
    for label in label_set:
        each_cluster = data[classfication_result == label, :]
        Wk = Wk + short_pair_wise_D(each_cluster) / (2.0 * each_cluster.shape[0])
    return Wk
def gap_statistic(X, B=10, K=range(1, 11), N_init=10):
    # 将输入数据集转换为数组
    X = np.array(X)
    # 生成B组参照数据
    shape = X.shape
    tops = X.max(axis=0)
    bots = X.min(axis=0)
    dists = np.matrix(np.diag(tops - bots))
    rands = np.random.random_sample(size=(B, shape[0], shape[1]))
    for i in range(B):
        rands[i, :, :] = rands[i, :, :] * dists + bots

    # 自定义0元素的数组，用于存储gaps、Wks和Wkbs
    gaps = np.zeros(len(K))
    Wks = np.zeros(len(K))
    Wkbs = np.zeros((len(K), B))
    # 循环不同的k值，
    for idxk, k in enumerate(K):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        classfication_result = k_means.labels_
        # 将所有簇内的Wk存储起来
        Wks[idxk] = compute_Wk(X, classfication_result)

        # 通过循环，计算每一个参照数据集下的各簇Wk值
        for i in range(B):
            Xb = rands[i, :, :]
            k_means.fit(Xb)
            classfication_result_b = k_means.labels_
            Wkbs[idxk, i] = compute_Wk(Xb, classfication_result_b)

    # 计算gaps、sd_ks、sk和gapDiff
    gaps = (np.log(Wkbs)).mean(axis=1) - np.log(Wks)
    sd_ks = np.std(np.log(Wkbs), axis=1)
    sk = sd_ks * np.sqrt(1 + 1.0 / B)
    # 用于判别最佳k的标准，当gapDiff首次为正时，对应的k即为目标值
    gapDiff = gaps[:-1] - gaps[1:] + sk[1:]

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制gapDiff的条形图
    plt.bar(np.arange(len(gapDiff)) + 1, gapDiff, color='steelblue')
    plt.xlabel('簇的个数')
    plt.ylabel('k的选择标准')
    plt.show()

# 调用寻找最佳k的函数，由图像看最佳的k
#k_SSE(data, 15)
#k_silhouette(data,15)
gap_statistic(data)


# kmeans =KMeans(n_clusters=3)
# kmeans.fit(data)
# data['cluster'] = kmeans.labels_
# print(data.cluster.value_counts())
