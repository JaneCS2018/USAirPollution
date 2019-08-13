#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
#import quandl
import math, datetime
import time
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.dates as mdates

pd.set_option('display.max_Rows', 15)

##A Good Article explains, https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR


from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
# set the max rows into 15 for display
# pd.set_option('display.height', 15)
pd.set_option('display.max_Rows', 15)

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split


# In[ ]:





# In[37]:


#read data
df = pd.read_csv("pollu1.csv", sep='|',usecols=["Date_Local","County","O3_AQI"])

df.head()


# ### plot the NO2 means in each day from 2000-01-01 to 2016-05-01
# df.plot(x = "Date_Local",
#        y = "NO2_Mean", 
#        figsize =(20, 10), 
#        title = 'NO2 Mean Value of USA Over Time', 
#        legend = False)

# In[45]:



df["O3_AQI"].max()
df=df.loc[df["County"] == "San Bernardino"]
df.head()


# In[42]:


data = data.groupby('Date_Local').mean()
data.head()


# In[51]:


#plot  in each day from 2000-01-01 to 2016-05-01
data.plot( y = "O3_AQI", figsize =(20, 10), title = 'O3 AQI of San Bernardino Over Time', legend = False)


# In[46]:


# above picture is too dense, we caculate the means of month then plot
df['Date_Local']=pd.to_datetime(df['Date_Local'],format='%Y-%m-%d')
df['YearMonth'] = pd.to_datetime(df['Date_Local']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
data2 = df.groupby('YearMonth').mean()
data2.head()


# In[48]:


# index is not a time formate and the timeline is disordered. Make index to column then tranform to a time formate
data2['Date']=data2.index
data2['Date']=pd.to_datetime(data2['Date'])
data2=data2.set_index('Date')
data2 = data2.sort_index() # order by index
data2.tail(15)


# In[49]:


# show Year-Month, hide day
df_period=data2.to_period('M')
df_period.head(15)


# In[50]:


df_period.plot(
       y = "O3_AQI", 
       figsize =(20, 10), 
       title = 'O3 AQI of San Bernardino Over Time', 
       legend = False)


# In[8]:


df_period.plot(
       y = "SO2_Mean", 
       figsize =(20, 10), 
       title = 'SO2 Mean Value of USA Over Time', 
       legend = False)


# In[9]:


df_period.plot(
       y = "O3_Mean", 
       figsize =(20, 10), 
       title = 'O3 Mean Value of USA Over Time', 
       legend = False)


# In[10]:


df_period.plot(
       y = "CO_Mean", 
       figsize =(20, 10), 
       title = 'CO Mean Value of USA Over Time', 
       legend = False)


# In[28]:


forecast_out = int(60) # predicting 60 months(5years) into future
df_period['Prediction'] = df_period[['O3_Mean']].shift(-forecast_out) 

df_period.head()


# In[29]:


X = np.array(df_period.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:] # set X_forecast equal to last 60
X = X[:-forecast_out] # remove last 60 from X
y = np.array(df_period['Prediction'])
y = y[:-forecast_out]


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)


# In[31]:


forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)


# In[32]:


df_period.iloc[-61,4]
for i in range(60):
    df_period.iloc[(i-60),4] = forecast_prediction[i];
df_period.tail()


# In[16]:


df_period['O3_Mean'].plot(figsize =(16, 8))
df_period['Prediction'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# In[33]:


clf.fit(X,y)
# Testing
confidence = clf.score(X, y)
print("confidence: ", confidence)


# In[34]:


forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)


# In[36]:



last_date=df_period.iloc[-1].name
print(last_date)


# In[37]:


last_date=datetime.datetime.strptime('2016-05', 
'%Y-%m') 
last_unix = last_date.timestamp()
one_month=86400*30
next_unix = last_unix + one_month


# In[38]:


for i in forecast_prediction:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_month #Seconds in a day
    df_period.loc[next_date] = [np.nan for _ in range(len(df_period.columns)-1)] + [i]


# In[39]:


df_period.tail()


# In[40]:


df_period['O3_Mean'].plot(figsize =(16, 8))
df_period['Prediction'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# In[ ]:




