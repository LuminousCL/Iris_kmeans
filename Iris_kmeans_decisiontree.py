#############鸢尾花数据集的决策树分析#############
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target, test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
predict_target = clf.predict(x_test)

print(sum(predict_target == y_test)) #预测结果与真实结果比对
print(metrics.classification_report(y_test,predict_target))
print(metrics.confusion_matrix(y_test,predict_target))

L1 = [n[0] for n in x_test]
L2 = [n[1] for n in x_test]
plt.scatter(L1,L2, c=predict_target,marker='x')
plt.title('DecisionTreeClassifier')
plt.show()
