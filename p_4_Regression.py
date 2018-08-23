# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

file_name = 'd:\\Users\\zhang\\Desktop\\数学建模之学校培训\\天津工业大学二轮模拟训练赛题\\赛题数据.xlsx'
data = pd.read_excel(file_name)

y_data = data.iloc[0:32155,14]
data.drop(columns=['student(n=32165)','school(0-159)','schtype(学校类型0-5)','gk_yw（高考语文）','gk_sx','gk_wy','gk_wl','gk_hx','gk_zf'],inplace=True)

X_train,X_test,y_train,y_test = train_test_split(data.values[0:32155],y_data,test_size=.15)

regressor = RandomForestRegressor(max_depth=9,n_estimators=50,max_features=.56)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
rf = pd.DataFrame(list(zip(y_pred,y_test)),columns=['y_pred','y_test'])


print(regressor.score(X_test,y_test))
print(regressor.score(X_train,y_train))

y_pred2 = regressor.predict(X_train)
rf2 = pd.DataFrame(list(zip(y_pred2,y_train)),columns=['y_pred2','y_train'])


regressor = RandomForestRegressor(max_depth=9,n_estimators=50,max_features=.56)
regressor.fit(data.values[0:32155],y_data)
y_pred = regressor.predict(data.values[32155:32165])
regressor.score(data.values[0:32155],y_data)

y_pred

result = regressor.predict(data.values[32155:32165])
print('预测得到的10名学生高考成绩：',result)

y_data = y_data.append(pd.Series(result.tolist()),ignore_index = True)
rank_ = y_data.rank(ascending=False)
temp = rank_.iloc[-10:32165].tolist()
print('预测的10个同学的排名情况：',temp)
def temp_(temp):
    for i in temp:
        if i<2501:
            yield 1
        elif i<10001:
            yield 2
        else:
            yield 0
print('可以上一类的标记为1，上二类的标记为2，不能上大学的标记为0：',list(temp_(temp)))


