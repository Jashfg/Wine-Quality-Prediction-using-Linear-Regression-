#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
data=pd.read_excel("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\wine  quality.xlsx")
df=pd.DataFrame(data)
df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.columns


# In[9]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.pairplot(data, vars=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'quality'], hue='quality')
plt.show()


# In[10]:


correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[12]:


plt.figure(figsize=(8, 6))
sns.countplot(x="quality",data=data)
plt.title("wine quality distribution")


# In[18]:


plt.figure(figsize=(18, 10))
sns.histplot(data["alcohol"],kde=True)
plt.title("Alcohol Content Distribution")


# In[19]:


df.head()


# In[20]:


features=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

x=df[features]
y=df["quality"]


# In[21]:


print(x.shape)
print(y.shape)


# In[25]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=19)


# In[26]:


print(x_train.shape)
print(x_test.shape)


# In[27]:


print(y_train.shape)
print(y_test.shape)


# In[29]:


from sklearn.linear_model import LinearRegression

regression=LinearRegression()
regression.fit(x_train,y_train)


# In[30]:


y_pred=regression.predict(x_test)
print("y_pred:",y_pred)


# In[32]:


y_test


# In[31]:


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("r2_score:",r2)


# In[ ]:




