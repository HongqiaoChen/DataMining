import pandas as pd
path = 'D:/test/10/Titanic.csv'
data = pd.read_table(path,sep=',')

#删除一部分没有用的列
data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
#检测是否有缺漏的项
IsNULL = data.isnull().sum(axis=0)
print(IsNULL)
#考虑到缺失的情况，对Embarked这一缺失但缺失较少的离散变量使用众数填充
data.fillna(value={'Embarked':data.Embarked.mode()[0]},inplace=True)
#考虑到缺失的情况，对Age这一缺失但缺失量较大的连续变量按照性别对客户的缺失年龄分组填充，
for i in data.Sex.unique():
    data.fillna(value = {'Age': int(data.Age[data.Sex == i].mean())}, inplace = True)
#检测是否还有缺失的情况
IsNULL = data.isnull().sum(axis=0)
print(IsNULL)
#对离散变量Pclass、Sex和Embarked变量进行哑变量处理（由字符型离散变量转为数值型离散变量）
print(data.Pclass)
data.Pclass = data.Pclass.astype('category')
print(data.Pclass)
dummy = pd.get_dummies(data[['Sex','Embarked','Pclass']])
data = pd.concat([data,dummy],axis=1)
data.drop(['Sex','Embarked','Pclass'],inplace=True,axis=1)
print(data)
pd.DataFrame.to_csv(data,'D:/test/ssss.csv')
