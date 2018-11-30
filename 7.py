import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import stats

# 本章节的思路：
# 1.怎么建线性回归模型
# 2.建好的的模型和模型的回归系数显著性如何
# 3.建好的模型能否用于实际生活使用（上述模型要成立，需要自变量和因变量满足那些要求）

# #一元线性回归模型
# #先画图
# income = pd.read_table('D:/test/7/Salary_Data.csv',sep=',')
# sns.lmplot(x='YearsExperience',y='Salary',data=income,ci=None)
# plt.show()
# #套用第三方库计算回归参数
# fit = sm.formula.ols('Salary ~ YearsExperience', data = income).fit()
# print(fit.params)

# #多元线性回归
Profit = pd.read_excel('D:/test/7/Predict to Profit.xlsx')
train,test = train_test_split(Profit,train_size=0.8,random_state=1234)
#state 变量为离散型的，需要把它转化为哑变量。将原变量放在C()中，表示将其当作分类（Category）变量处理
model = sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + C(State)', data=train).fit()
#所得到的参数解释：以RD_Spend(连续型)和State（离散型，转化为哑变量）
#RD_Spend预测的参数是0.803487，表示成本每增加一元，利润多0.8
#State T.Florida表示 ，在以California为基准的情况下，其他条件不变，利润多了927.394424
print(model.params)
test_X = test.drop(labels='Profit',axis=1)
pred = model.predict(exog = test_X)
result = pd.DataFrame({'Prediction':pred,'Real':test.Profit})
print(result)

# 对线性回归模型显著性检验——F检验
# 直接计算
# ybar = train['Profit'].mean()
# p = model.df_model
# n = train.shape[0]
# RSS = np.sum((model.fittedvalues-ybar)**2)
# ESS = np.sum(model.resid**2)
# F = (RSS/p)/(ESS/(n-p-1))
# print(F)
# 或者直接使用属性.fvalue
#print(model.fvalue)
# 根据F检验的结果，对线性回归模型显著性下个结论(一般置信度置为0.95,即置信水平为0.05)
# 通过了模型的显著性检验之后，只能说明关于因变量的线性组合是合理的，但并不能说明没个自变量队因变量都有显著意义
#F_Theroy = stats.ppf(q=0.95,dfn=p,dfd=n-p-1)
#print('如果F检验的结果远远大于F_Theroy,则认为多元线性回归模型显著。现在F=%f，F_Theroy=%f。'%(F,F_Theroy))

# 为了说明因变量有显著性意义，还需要队回归系数进行显著性检验——即t检验
# 直接调用model.summary()查看。t检验的结果显示在第二个表中。我们一般认为置信水平应该为0.05，当P>|t|的结果大于0.05的时候我们认为他拒绝了原来的假设
# 即在本例中除了Intercept和RD_Spend都应该视为不是影响利润的重要因素
#print(model.summary())

#在建立好模型之后，需要对模型再度进行检验
# 正态性检验
# 直方图法
# 导入第三方模块
# import scipy.stats as stats
# # 中文和负号的正常显示
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 绘制直方图
# sns.distplot(a = Profit_New.Profit, bins = 10, fit = stats.norm, norm_hist = True,
#              hist_kws = {'color':'steelblue', 'edgecolor':'black'},
#              kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
#              fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
# # 显示图例
# plt.legend()
# # 显示图形
# plt.show()
#
#
# # 残差的正态性检验（PP图和QQ图法）
# pp_qq_plot = sm.ProbPlot(Profit_New.Profit)
# # 绘制PP图
# pp_qq_plot.ppplot(line = '45')
# plt.title('P-P图')
# # 绘制QQ图
# pp_qq_plot.qqplot(line = 'q')
# plt.title('Q-Q图')
# # 显示图形
# plt.show()
#
#
# # 导入模块
# import scipy.stats as stats
# stats.shapiro(Profit_New.Profit)
#
# # 生成正态分布和均匀分布随机数
# rnorm = np.random.normal(loc = 5, scale=2, size = 10000)
# runif = np.random.uniform(low = 1, high = 100, size = 10000)
# # 正态性检验
# KS_Test1 = stats.kstest(rvs = rnorm, args = (rnorm.mean(), rnorm.std()), cdf = 'norm')
# KS_Test2 = stats.kstest(rvs = runif, args = (runif.mean(), runif.std()), cdf = 'norm')
# print(KS_Test1)
# print(KS_Test2)

# 多重共线性检验
# 导入statsmodels模块中的函数
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 自变量X(包含RD_Spend、Marketing_Spend和常数列1)
X = sm.add_constant(Profit.ix[:,['RD_Spend','Marketing_Spend']])
# 构造空的数据框，用于存储VIF值
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# 如果存在变量的vif值大于了10这说明存在多重共线性的情况。不应该考虑重新选择模型（岭回归模型或LASSO模型）
print(vif)
