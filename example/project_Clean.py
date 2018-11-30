import pandas as pd
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale

def ISNULL(data):
    ISNULL = data.isnull().sum(axis=0)
    print(ISNULL)

path = 'D:/test/10/Titanic.csv'
data = pd.read_table(path,sep=',')
# 删除无用的列
data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
# 检测是否有NAN
ISNULL(data)
# 对存在NAN的地方用平均数或众数回填
data.fillna(value={'Embarked':data.Embarked.mode()[0]},inplace=True)
data.fillna(value={'Age':int(data.Age.mean())},inplace=True)

# 显示此刻data中的各列的数据类型
# print(data.info())

# #将类别变量设为哑变量
# 先将数值型类别变量转化为category格式
data.Pclass = data.Pclass.astype('category')
# 将category格式和objec格式的类别变量转化为哑变量
dummy = pd.get_dummies(data[['Pclass','Sex','Embarked']])
# 用哑变量代替原来的变量
data = pd.concat([data,dummy],axis=1)
data.drop(['Pclass','Sex','Embarked'],axis=1,inplace=True)

# #划分因变量和自变量
prediction = data.columns[1:]
X = data[prediction]
Y = data['Survived']
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=1234)
# 归一化自变量(一般不需要)
# X = minmax_scale(X)
# print(X)

# 检测因变量是否存在失衡
counts = data.Survived.value_counts()
print((counts[1]/(counts[0]+counts[1])))
# 若失衡
# over_sample = SMOTE(random_state=1234)
# X_train,Y_train = over_sample.fit_sample(X_train,Y_train)

pd.DataFrame.to_csv(data,'D:/test/sss.csv')




