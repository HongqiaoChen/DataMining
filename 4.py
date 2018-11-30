import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
def linear_regression(X,Y):
    X_trans_X_inverse = np.linalg.inv(np.dot(np.transpose(X),X))
    beat = np.dot(np.dot(X_trans_X_inverse,np.transpose(X)),Y)
    return beat

# 读取txt文件
# stu_score = np.loadtxt('D:/test/stu_score.txt',dtype=int,skiprows=1)
# print(stu_score)
# print(stu_score.shape)
# print(stu_score.dtype)
# print(stu_score.size)
# print(len(stu_score))
# 把数组中小于80的用0标记，大80的用1标记
# print(np.where(stu_score>80,1,0))
# 把数组中小于80的标记为0，其他的维持原数组的值
# print(np.where(stu_score<80,0,stu_score))
# 降维
# print(stu_score.reshape(-1,order='F'))

# 维度不同的数组相加
# 维数不一致，但是末尾的维度一致：相当 arr2中第一个维度中每一个都是加上了arr1
# arr1 = np.arange(1,13).reshape(3,4)
# print(arr1)
# arr2 = np.arange(60).reshape(5,3,4)
# print(arr2)
# print(arr2+arr1)
# 最后一个维度为1，倒数的二个维度相同
# arr3 = np.arange(60).reshape(5,4,3)
# arr4 = np.arange(4).reshape(4,1)
# print(arr3+arr4)

# 得到一个方正的特征值与特征向量组成的原组
# arr5 = np.array([[1,2,5],[3,6,8],[4,7,9]])
# print(np.linalg.eig(arr5))

# 计算多元线性回归模型的解的回归系数
# X = np.array([[1,1,4,3],[1,2,7,6],[1,2,6,6],[1,3,8,7],[1,2,5,8],[1,3,7,5],[1,6,10,12],[1,5,7,7],[1,6,3,4],[1,5,7,8]])
# Y = np.array([3.2,3.8,3.7,4.3,4.4,5.2,6.7,4.8,4.2,5.1])
# beat = linear_regression(X,Y)

# 解方程组R=V的非齐次线性方程组
# A = np.array([[3,2,1],[2,3,1],[1,2,3]])
# b = np.array([39,34,26])
# X = np.linalg.solve(A,b)
# print(X)

# 计算向量的范数(向量的二阶范数就是向量的模)
# arr7 = np.array([1,3,5,7,9,10,-12])
# res1 = np.linalg.norm(arr7,ord=1)
# res2 = np.linalg.norm(arr7,ord=2)
# res3 = np.linalg.norm(arr7,ord=np.inf)
# print(res1)
# print(res2)
# print(res3)

#基于np取随机数，取得的随机数是按特定的分布的
np.random.seed(123)
rn1 = np.random.normal(loc= 0,scale= 1,size=1000)
rn2 = np.random.normal(loc= 0,scale= 3,size=1000)
rn3 = np.random.normal(loc=2,scale=1,size=1000)
plt.style.use('ggplot')
#seaborn中的displot函数 参数fit 提供一个带对比的图样
sns.distplot(rn1,fit=stats.norm,hist=False,kde=True,fit_kws={'color':'red','label':'u=0,s=1','linestyle':'-'})
sns.distplot(rn2,fit=stats.norm,hist=False,kde=True,fit_kws={'color':'blue','label':'u=0,s=3','linestyle':'-.'})
plt.legend() #显示图例
plt.show()   #显示图形
