#############打印鸢尾花前五行数据(引用库)#############
from sklearn import datasets

iris = datasets.load_iris()

# data对应了样本的4个特征，150行4列
print(iris.data.shape)

# 显示样本特征的前5行
print(iris.data[:5])

# target对应了样本的类别（目标属性），150行1列
print(iris.target.shape)

# 显示所有样本的目标属性
print(iris.target)