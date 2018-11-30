import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt


#构造series，可以通过list，数组，元组等实现
# gdp1 = pd.Series([2.8,3.01,8.99,8.59,5.18])
# gdp2 = pd.Series({'北京':'2.8','上海':'3.01','广东':'8.99','江苏':'8.59','浙江':'5.18'})
# gdp3 = pd.Series({'北京':2.8,'上海':3.01,'广东':8.99,'江苏':8.59,'浙江':5.18})
# print(gdp1)
# print(gdp2)
# print(gdp3)
# print(gdp1[[0,3,4]])
# print(gdp2[[0,3,4]])
# print(gdp3[['北京','江苏','浙江']])
# 对series对象可以使用np的相关方法
# 对series对象，统计一般采用pandas库的相关方法，数学计算一般采用numpy库的相关方法
# print(np.log(gdp3))
# print(np.mean(gdp1))

#由字典使用DataFrame函数构造数据框
# df = pd.DataFrame({'姓名':['h','q','c'],'年龄':[20,1,33],'性别':['男','女','男']})
# print(df)

#由Excel生成DataFrame，将生日合并且将相关的信息提取出来
# user_income = pd.read_table('D:/test/5/data_test01.txt',skiprows=2,skipfooter=3,sep=',',encoding='utf-8',thousands='&',comment='#',parse_dates={'birthday':[0,1,2]},engine='python')
# print(user_income)

# child_cloth = pd.read_excel('D:/test/5/data_test02.xlsx','Sheet1',header=None,converters={0:str},names={'ID':0,'Name':1,'Color':2,'Price':3})
# print(child_cloth)
# child_cloth = pd.read_excel('D:/test/5/data_test06.xlsx','Sheet1',converters={0:str})
# print(child_cloth)

# 从数据库中读取数据生成DataFrame
# conn = pymysql.connect(host='localhost',user = 'root',password = 'cmr070414',database='dns',port=3306,charset='utf8')
# dns = pd.read_sql('select * from dns',conn)
# conn.close()
# print(dns)

# sec_cars = pd.read_table('D:/test/5/sec_cars.csv',delimiter=',',encoding='utf-8')
# print(sec_cars.head())
# print(sec_cars.shape)
# print(sec_cars.dtypes)
# 将年月格式改为 ****-**-**的格式
# sec_cars.Boarding_time = pd.to_datetime(sec_cars.Boarding_time,format='%Y年%m月')
# 将字符型的New_price改为浮点数类型(如果数据完整的话：不包含暂无这些邪教)
# sec_cars.New_price = sec_cars.New_price.str[0:-2].astype(np.float)
# print(sec_cars)
# 查看sec_cars中的数据描述性统计情况（平均值，最大，最小值，方差等）
# a = sec_cars.describe()
# print(a)

# info = pd.read_excel('D:/test/5/data_test03.xlsx')
# 将手机号中间4位改为*
# info.tel = info.tel.astype('str')
# for i in range(len(info.tel)):
#     a = list(info.tel[i])
#     for j in range(3,7):
#         a[j] = '*'
#     info.tel[i] = ''.join(a)
# 把年龄的格式标准化
# info.birthday = pd.to_datetime(info.birthday,format='%Y/%m/%d')
# print(info.birthday)
# 计算年龄
# info['age'] = pd.datetime.today().year - info.birthday.dt.year

# 删除重复值,要真正改变原df需要加上inplace = true
# df = pd.read_excel('D:/test/5/data_test04.xlsx')
# df.drop_duplicates(inplace=True)
# print(df.duplicated())

# 缺失值处理,要真正改变原df需要加上inplace = true
# df = pd.read_excel('D:/test/5/data_test05.xlsx')
# print(any(df.isnull()))#确定df中是否有null
# 删除法
# a = df.dropna()
# print(a)
# 替换法
# 向前替换法
# print(df.fillna(method='ffill'))
# 向后替换法
# print(df.fillna(method='bfill'))
# 常数替换法
# print(df.fillna(value=0))
# 统计值替换法，本例子中用众数替换性别，用平均数替换年龄，用中位数替换收入
# print(df.fillna(value={'gender':df.gender.mode()[0],'age':df.age.mean(),'income':df.income.median()}))

#异常值处理

# sunspots = pd.read_table('D:/test/5/sunspots.csv',sep=',')
# print(sunspots)
# #标准差异常检测法
# xbar = sunspots.counts.mean()
# xstd = sunspots.counts.std()
# print(xbar)
# print(xstd)
# #标准差法异常值上限检验，若为true则有异常
# up = any(sunspots.counts>xbar+2*xstd)
# #标准差法异常值下限检验，若为true则有异常
# down = any(sunspots.counts<xbar-2*xstd)
# print(up)
# print(down)
# #箱线图异常值判别法
# Q1 = sunspots.counts.quantile(q=0.25)
# print(Q1)
# Q2 = sunspots.counts.quantile(q=0.75)
# print(Q2)
# IQR = Q2-Q1
# # 有异常则up为true
# up = any(sunspots.counts>Q2+1.5*IQR)
# print(up)
# # 有异常则down为true
# down = any(sunspots.counts<Q1-1.5*IQR)
# print(down)
#画图看偏移
# plt.style.use('ggplot')
# sunspots.counts.plot(kind='hist',bins=30,density=True)
# sunspots.counts.plot(kind='kde')
# plt.show()

# 把异常值用合理值代替
# UL = Q2 +1.5*IQR
# DL = Q1 -1.5*IQR
# relpace_value_up = sunspots.counts[sunspots.counts<UL].max()
# sunspots.counts[sunspots.counts>UL] = relpace_value_up
# relpace_value_down = sunspots.counts[sunspots.counts>DL].min()
# sunspots.counts[sunspots.counts<DL] = relpace_value_down
# print(sunspots.counts.describe())

# ix将df1获取数据的子集
# df1 = pd.DataFrame({'name':['zhangsan','lisi','waner','dingyi','hongqiaochen'],'gender':['M','F','F','F','M'],'age':[20,22,25,13,22]},columns=['name','gender','age'])
# print(df1)
# print(df1.ix[1:3,[0,2]])
# print(df1.ix[1:3,['name','age']])

# diamonds = pd.read_table('D:/test/5/diamonds.csv',sep=',')
# print(diamonds)
# print(pd.pivot_table(data=diamonds,index='color',values='price',margins=True,margins_name='总计'))
# #aggfunc=np.size表示统计value=x的结果x的size。如果没有aggfunc则默认为values值的平均值
# print(pd.pivot_table(data=diamonds,index='clarity',columns='cut',values='x',aggfunc=np.size,margins=True,margins_name='总计'))

# df1 = pd.DataFrame({'name':['zhangsan','lisi','waner'],'age':[10,12,18],'gender':['M','F','M']})
# df2 = pd.DataFrame({'name':['dingyi','zhaowu'],'age':[15,16],'gender':['F','F']})
# df3 = pd.DataFrame({'income':[1000,2000]})
#
# print(pd.concat([df1,df2],keys=['df1','df2'],axis=0))
# print(pd.concat([df2,df3],axis=1))