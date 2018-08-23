# -*- coding: utf-8 -*-
import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot  as plt

file_name = 'd:\\Users\\zhang\\Desktop\\数学建模之学校培训\\天津工业大学二轮模拟训练赛题\\data.xlsx'
# file_name_1 = 'd:\\Users\\zhang\\Desktop\\数学建模之学校培训\\天津工业大学二轮模拟训练赛题\\data_1.xlsx'
# file_name_2 = 'd:\\Users\\zhang\\Desktop\\数学建模之学校培训\\天津工业大学二轮模拟训练赛题\\data_2.xlsx'

data = pd.read_excel(file_name,header=None)
y_data = data.iloc[:,12]
X_train,X_test,y_train,y_test = train_test_split(data.values[:,0:-1],y_data,test_size=.15)
clf = RandomForestClassifier(max_depth=9,n_estimators=100,max_features=.56)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
rf = pd.DataFrame(list(zip(y_pred,y_test)),columns=['y_pred','y_test'])
print(clf.score(X_test,y_test))
print(clf.score(X_train,y_train))

file_name = 'd:\\Users\\zhang\\Desktop\\数学建模之学校培训\\天津工业大学二轮模拟训练赛题\\赛题数据.xlsx'
# file_name_1 = 'd:\\Users\\zhang\\Desktop\\数学建模之学校培训\\天津工业大学二轮模拟训练赛题\\data_1.xlsx'
# file_name_2 = 'd:\\Users\\zhang\\Desktop\\数学建模之学校培训\\天津工业大学二轮模拟训练赛题\\data_2.xlsx'

data_ = pd.read_excel(file_name)
data_.drop(columns=['student(n=32165)','school(0-159)','schtype(学校类型0-5)','gk_yw（高考语文）','gk_sx','gk_wy','gk_wl','gk_hx','gk_zf'],inplace=True)

clf = RandomForestClassifier(max_depth=9,n_estimators=100,max_features=.56)
clf.fit(data.values[:,0:-1],y_data)
y_pred = clf.predict(data_.values[32155:32165])
#clf.score(data.values[:,0:-1],y_data)
print(clf.predict(data_.values[32155:32165]))

#plt.rcParams['font.sas-serig']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.barh(range(12),clf.feature_importances_,color='g')
xx = ['中考语文','中考数学','中考外语','中考物理','中考化学','中考总分',
'学校语文','学校数学','学校外语','学校物理','学校化学','学校总分']
plt.yticks(range(12),xx,fontproperties='SimHei', size=50)
plt.title('是否上大学和上哪类大学与各成绩之间影响大小关系',fontproperties='SimHei', size=50)