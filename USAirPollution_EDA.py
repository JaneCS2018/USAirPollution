#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import date
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from pandas.plotting import scatter_matrix


# In[2]:


data = pd.read_csv('/Users/xiangliu/Desktop/CSC560 Data/pollution_us_2000_2016.csv')
data.shape


# In[3]:


data.head(3)


# In[4]:


le=data['CO AQI']
le[le.isnull()]


# In[5]:


##Look Into each value
data=data.dropna()
data.to_csv("/Users/xiangliu/Desktop/CSC560 Data/pollution_AQI.csv",index=True,sep=',')


# In[6]:


data = pd.read_csv('/Users/xiangliu/Desktop/CSC560 Data/pollution_AQI.csv')
data.shape


# In[7]:


data.head(3)


# In[8]:


data.groupby(['State']).count()
#Found there are 5 states no there(Montana, Nebraska, Mississippi, West virgina, Vermont)


# In[9]:


le=data['CO AQI']
le[le.isnull()]


# In[10]:


le=data['SO2 AQI']
le[le.isnull()]


# In[11]:


le=data['NO2 AQI']
le[le.isnull()]


# In[12]:


le=data['O3 AQI']
le[le.isnull()]


# In[13]:


le=data['SO2 Mean']
le[le.isnull()]


# In[14]:


le=data['CO Mean']
le[le.isnull()]


# In[15]:


le=data['NO2 Mean']
le[le.isnull()]


# In[16]:


le=data['O3 Mean']
le[le.isnull()]


# In[17]:


data.head(3)


# In[18]:


data['SO2 Mean'].mean()


# In[19]:


sns.boxplot(y='SO2 Mean', data=data.loc[(data['SO2 Mean'] < 10)])


# In[20]:


fig, ax = plt.subplots(figsize=(25, 10))
sns.boxplot(x='State', y='SO2 Mean', data=data.loc[(data['SO2 Mean'] < 10)])
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.show();


# In[21]:


data['CO Mean'].mean()


# In[22]:


sns.boxplot(y='CO Mean', data=data.loc[(data['CO Mean'] < 1)])


# In[23]:


sns.boxplot(y='CO AQI', data=data)


# In[24]:


data['CO AQI'].max()


# In[25]:


county=data.loc[(data['CO AQI'] > 100) & (data['CO AQI'] < 300)]
county.groupby(['State']).count()


# In[26]:


county.groupby(['County']).head(20)


# In[27]:


fig, ax = plt.subplots(figsize=(25, 10))
sns.boxplot(x='State', y='CO Mean', data=data.loc[(data['CO Mean'] < 1)])
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.show();


# In[28]:


data['NO2 Mean'].mean()


# In[29]:


sns.boxplot(y='NO2 Mean', data=data.loc[(data['NO2 Mean'] < 100)])


# In[30]:


fig, ax = plt.subplots(figsize=(25, 10))
sns.boxplot(x='State', y='NO2 Mean', data=data.loc[(data['NO2 Mean'] < 60)])
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.show();


# In[31]:


data['O3 Mean'].mean()


# In[32]:


sns.boxplot(y='O3 Mean', data=data.loc[(data['O3 Mean'] <1)])


# In[33]:


###Correlation Coefficient
col = ['SO2 Mean',
       'NO2 Mean',
       'CO Mean',
       'O3 Mean',
       'SO2 AQI',
       'NO2 AQI',
       'CO AQI',
       'O3 AQI'
       ]
corr = data.loc[(data['SO2 Mean'] <10)&(data['CO Mean'] <1)&(data['NO2 Mean'] <40)&(data['O3 Mean'] <0.08)][col].dropna().corr()
plt.figure(figsize = (10,10))
sns.set(font_scale=1)
sns.heatmap(corr, cbar = True, annot=True, square = True, fmt = '.2f', xticklabels=col, yticklabels=col)
plt.show();


# In[34]:


# scatter = data.loc[(data['SO2 Mean'] <50)&(data['CO Mean'] <10)&(data['NO2 Mean'] <100)&(data['O3 Mean'] <1)][col]
# scatter_matrix(scatter, figsize =(20,20), color='lightblue')


# In[35]:


#data.groupby(['State']).sort_values("SO2_Mean").plot.bar()
#sns.barplot(x='State', y="SO2_Mean", data=data)


# In[42]:


sns.set(font_scale=1)
sort = data.loc[(data['SO2 Mean'] <50)]                    .groupby('State')['SO2 Mean']                    .mean()                    .sort_values(ascending=False)                    .index

sns.barplot(y='SO2 Mean', x='State', data=data, order=sort, palette="Blues_d")
sns.set_context({"figure.figsize": (30, 8)})
plt.xticks(rotation=45)
plt.show()


# In[37]:


sns.set(font_scale=1.5)
sort = data.loc[(data['CO Mean'] <10)]                    .groupby('State')['CO Mean']                    .mean()                    .sort_values(ascending=False)                    .index

sns.barplot(y='CO Mean', x='State', data=data, order=sort, palette="Blues_d")
sns.set_context({"figure.figsize": (30, 8)})
plt.xticks(rotation=45)
plt.show()


# In[38]:


sns.set(font_scale=1.5)
sort = data.loc[(data['NO2 Mean'] <50)]                    .groupby('State')['NO2 Mean']                    .mean()                    .sort_values(ascending=False)                    .index

sns.barplot(y='NO2 Mean', x='State', data=data, order=sort, palette="Blues_d")
sns.set_context({"figure.figsize": (30, 8)})
plt.xticks(rotation=45)
plt.show()


# In[39]:


sns.set(font_scale=1.5)
sort = data.loc[(data['O3 Mean'] <0.08)]                    .groupby('State')['O3 Mean']                    .mean()                    .sort_values(ascending=False)                    .index

sns.barplot(y='O3 Mean', x='State', data=data, order=sort, palette="Blues_d")
sns.set_context({"figure.figsize": (30, 8)})
plt.xticks(rotation=45)
plt.show()


# In[40]:


#Build training model
# scatter = data.loc[(data['SO2 Mean'] <50)&(data['CO Mean'] <10)&(data['NO2 Mean'] <100)&(data['O3 Mean'] <1)][col]
# scatter_matrix(scatter, figsize =(20,20), color='lightblue')

