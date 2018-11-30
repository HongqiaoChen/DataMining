import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #用于得到训练集和测试集
from sklearn.neighbors import KNeighborsClassifier #用于初始化KN的参数
from sklearn.model_selection import GridSearchCV #KN的网格收索函数
from sklearn import metrics #用于计算auc，评价模型

#对数据进行标准化处理——一般涉及到对重复值，对缺失值，对异常值的处理
income = pd.read_excel('D:/test/income.xlsx')
#对缺失值的处理。对相应的项，用统计值加以填补。对于离散型的，一般使用众数（.mode）的方式；对于连续型的，一般使用平均数（.mean）的方式。
print(income.apply(lambda x:np.sum(x.isnull())))
income.fillna(value={'workclass':income.workclass.mode()[0],'occupation':income.occupation.mode()[0],'native-country':income['native-country'].mode()[0]},inplace=True)
#对重复值的处理。一般采取删除重复项的措施。
print(any(income.duplicated()))
income.drop_duplicates(inplace=True)
#对异常值的处理。一般先采用箱线图判别法（或n个标准差法）判断是否存在异常值，然后再对存在的异常值进行处理。此处略-。-！

#纵观全局的数字性统计指标
# print(income.describe())
#纵观全局的非数字性指标
# print(income.describe(include=['object']))

#画图分析
# #针对连续性随机变量，往往作核密度图以估计
# plt.style.use('ggplot')
# # 设置多图形的组合
# fig, axes = plt.subplots(2, 1)
# # 绘制不同收入水平下的年龄核密度图
# income.age[income.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[0], legend = True, linestyle = '-')
# income.age[income.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[0], legend = True, linestyle = '--')
# # 绘制不同收入水平下的周工作小时数核密度图
# income['hours-per-week'][income.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[1], legend = True, linestyle = '-')
# income['hours-per-week'][income.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[1], legend = True, linestyle = '--')
# # 显示图形
# plt.show()
# #针对离散型随机变量
# #本行是为了得到race和income的聚类，统计的是个数（np.size），其中loc[:,'age']只是用于计量的变量。即从开始到结束的age有多少个
# race = pd.DataFrame(income.groupby(by = ['race','income']).aggregate(np.size).loc[:,'age'])
# # 重设行索引
# race = race.reset_index()
# # 变量重命名
# race.rename(columns={'age':'counts'}, inplace=True)
# # 排序
# race.sort_values(by = ['race','counts'], ascending=False, inplace=True)
# # 构造不同收入水平下各家庭关系人数的数据
# relationship = pd.DataFrame(income.groupby(by = ['relationship','income']).aggregate(np.size).loc[:,'age'])
# relationship = relationship.reset_index()
# relationship.rename(columns={'age':'counts'}, inplace=True)
# relationship.sort_values(by = ['relationship','counts'], ascending=False, inplace=True)
# # 设置图框比例，并绘图
# plt.figure(figsize=(9,5))
# sns.barplot(x="race", y="counts", hue = 'income', data=race)
# plt.show()
# plt.figure(figsize=(9,5))
# sns.barplot(x="relationship", y="counts", hue = 'income', data=relationship)
# plt.show()

# 对离散的变量进行编码
for each in income.columns:
    if income[each].dtype =='object':
        income[each] = pd.Categorical(income[each]).codes
#删除没有统计意义或重复的列
income.drop(['education','fnlwgt'],axis=1,inplace=True)


#对数据集进行拆分
X_train,X_test,Y_train,Y_test = train_test_split(income.loc[:,'age':'native-country'],income['income'],train_size=0.75,random_state=4321)
print(X_train.shape)

#构建默认参数的kn模型
kn = KNeighborsClassifier()
kn.fit(X_train,Y_train)
print(kn)

# 采用的是网格收索法，训练kn模型的参数
# k_options = list(range(1,12))
# parameters = {'n_neighbors':k_options}
# grid_kn = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=parameters,cv=10,scoring='accuracy')
# grid_kn.fit(X_train,Y_train)
# print(grid_kn.best_params_)

#评价kn模型，由于本子慢，此处仅对于默认的kn模型进行评价
#混淆矩阵法
# kn_pred = kn.predict(X_test)
# print(pd.crosstab(kn_pred,Y_test))
# print(kn.score(X_train,Y_train))
# print(kn.score(X_test,Y_test))
#AUC法
fpr,tpr,_ = metrics.roc_curve(Y_test,kn.predict_proba(X_test)[:,1])
plt.plot(fpr,tpr,linestyle='-',color='red')
plt.text(0.6,0.4,'AUC=%.3f'%metrics.auc(fpr,tpr))
print('AUC=%.3f'%metrics.auc(fpr,tpr))
plt.show()


