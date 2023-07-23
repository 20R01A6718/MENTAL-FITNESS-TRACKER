#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas numpy matplotlib seaborn scikit-learn plotly.express


# In[5]:


import pandas as pd
import numpy as np


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[7]:


ds1 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')


# In[8]:


ds2 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder - Copy.csv")


# In[14]:


ds1.head(3)


# In[15]:


ds2.head(3)


# In[16]:


data = pd.merge(ds1, ds2)
data.head()


# In[17]:


data.isnull().sum()


# In[18]:


data.drop('Code', axis=1, inplace=True)


# In[19]:


data.size


# In[20]:


data.shape


# In[21]:


data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)
data.head()


# In[22]:


plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Greens')
plt.plot()


# In[23]:


sns.jointplot(data,x="Schizophrenia",y="mental_fitness",kind="reg",color="m")
plt.show()


# In[24]:


sns.jointplot(data,x='Bipolar_disorder',y='mental_fitness',kind='reg',color='blue')
plt.show()


# In[25]:


sns.pairplot(data,corner=True)
plt.show()


# In[27]:


mean = data['mental_fitness'].mean()
mean


# In[28]:


fig = px.pie(data, values='mental_fitness', names='Year')
fig.show()


# In[29]:


fig=px.bar(data.head(10),x='Year',y='mental_fitness',color='Year',template='ggplot2')
fig.show()


# In[30]:


fig = px.line(data, x="Year", y="mental_fitness", color='Country', markers=True, color_discrete_sequence=['green', 'orange', 'purple'], template='plotly_dark')
fig.show()


# In[31]:


df=data.copy()
df.head()


# In[32]:


df.info()


# In[33]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])


# In[34]:


X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']


# In[35]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)


# In[36]:


print("xtrain: ", xtrain.shape)
print("xtest: ", xtest.shape)
print("ytrain: ", ytrain.shape)
print("ytest: ", ytest.shape)


# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)

# model evaluation for training set
ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = lr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[38]:


# RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)
# model evaluation for training set
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)
print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
# model evaluation for testing set
ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)
print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))



# In[39]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))


# In[40]:


print('R2 score is {}'.format(r2))


# In[ ]:


import numpy as np
from sklearn.preprocessing import LabelEncoder

np.random.seed(0)

print("Welcome to Mental Fitness Tracker!\nFill in the details to check your mental fitness!")

l = LabelEncoder()
country = l.fit_transform([input('Enter Your country Name:')])

year = int(input("Enter the Year:"))

schi = float(input("Enter your Schizophrenia rate in % (if not applicable, enter 0):")) * 100
bipo_dis = float(input("Enter your Bipolar disorder rate in % (if not applicable, enter 0):")) * 100
eat_dis = float(input("Enter your Eating disorder rate in % (if not applicable, enter 0):")) * 100
anx = float(input("Enter your Anxiety rate in % (if not applicable, enter 0):")) * 10
drug_use = float(input("Enter your Drug Usage rate per year in % (if not applicable, enter 0):")) * 100
depr = float(input("Enter your Depression rate in % (if not applicable, enter 0):")) * 10
alch = float(input("Enter your Alcohol Consuming rate per year in % (if not applicable, enter 0):")) * 100

# Calculate mental fitness score
mental_fitness = 100 - (schi + bipo_dis + eat_dis + anx + drug_use + depr + alch) / 7

print(f"\nYour mental fitness score is: {mental_fitness:.2f}%")


# In[ ]:





# In[ ]:




