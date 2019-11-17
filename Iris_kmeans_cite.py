#############K-means-鸢尾花聚类(引用)#############
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # from sklearn import datasets
from sklearn.datasets import load_iris

iris = load_iris()
print(iris)
X = iris.data[:, 2:]  # 取后两个维度
print(X)

# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签

# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="pink", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='x', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()