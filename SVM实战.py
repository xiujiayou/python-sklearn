#!/usr/bin/env python
# coding: utf-8

# In[34]:


#引入sklearn的工具包
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[35]:


data=pd.read_csv('svmdata.csv')


# In[17]:


data.head()


# In[18]:


data.describe()


# In[19]:


#因为数据集列比较多，需要把df的列都显示出来
#显示所有列----pd.set_option('display.max_columns', None)
#显示所有行----pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns',None)
print(data.columns)
print(data.head(5))
print(data.describe())


# In[20]:


#将特征字段分成3段
features_mean=list(data.columns[2:12])
features_se=list(data.columns[12:22])
features_worst=list(data.columns[22:32])


# In[21]:


#数据清洗
#id列没有用，删除
data.drop('id',axis=1,inplace=True)
#将B良性替换成0，恶性替换成1
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})


# In[22]:


data.head()


# In[23]:


#接下来进行特征字段的筛选，首先需要观察下features_mean各个变量之间的关系。可以用corr（）的函数

#将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'],label="Count")
plt.show()
#用热力图呈现 features_mean字段之间的相关性
corr=data[features_mean].corr()
plt.figure(figsize=(14,14))
#annot=True 显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()


# 图中颜色越浅表示相关性越大。所以我们能发现：radius_mean、perimeter_mean，area_mean这三个相关性大，而compactness_mean、concavity_mean，concave_points_mean这三个特征相关性大。

# 下面来进行特征选择
# 特征选择的目的是降维，用少量的特征代表数据的特性，这样也可以增强分类器的泛化能力，避免数据过拟合。
# mean/se/worst这三组特征是对同一组内容的不同度量。我们可以只保留mean这组特征。
# 另外，在mean这组特征中，radius_mean,perimeter_mean,area_mean这三个属性相关性大，
# 而compactness_mean、concavity_mean，concave_points_mean这三个特征相关性大。
# 我们可以分别从这2类中选择一个属性作为代表，比如radius_mean和compactness_mean
# 
# 那么我们就可以把原来的10个属性缩减为6个属性
# 

# In[24]:


#特征选择
features_remain=['radius_mean','texture_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean']


# #准备训练集和测试集
# #30%为测试集，70%为训练集
# 

# In[29]:


#记得引入---from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)
train_X=train[features_remain]
train_y=train['diagnosis']
test_X=test[features_remain]
test_y=test['diagnosis']


# 训练之前，需要对数据进行规范化，让数据在同一个量级上，避免因为维度问题造成数据误差。

# In[38]:


#采用z-score 规范化数据，保证每个特征维度的数据均值为0，方差为1
#记得引入---from sklearn import preprocessing
ss=preprocessing.StandardScaler()
train_X=ss.fit_transform(train_X)
test_X=ss.transform(test_X)


# In[43]:


from sklearn.metrics import accuracy_score
#创建svm分类器
model=svm.SVC()
#用训练集做训练
model.fit(train_X,train_y)
#用测试集做预测
prediction=model.predict(test_X)
print('准确率： ',accuracy_score(prediction,test_y))


# In[ ]:




