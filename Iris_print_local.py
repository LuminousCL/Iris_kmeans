#############打印鸢尾花前五行数据(本地)#############
import pandas as pd

data=pd.read_csv('iris.csv')
#iris.data数据与程序文件存放在同一目录下

print(data.head(5))
#可以查看一下前5行数据，检查是否读取正确

attributes=data[['sl','sw','pl','pw']]
#前四列属性简化为sl，sw，pl，pw

types=data['type']
#第5列属性为鸢尾花的类别
