#!/usr/bin/env python
# coding: utf-8

# ### Multiple Regression

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


### IMPORT THE DATASET


# In[11]:


df=pd.read_csv('mtcars.csv')


# In[12]:


df.shape


# In[13]:


df


# In[20]:


df.corrwith(df['mpg'])


# In[15]:


salary=pd.read_csv('Salary_Data.csv')


# In[16]:


salary.corr()


# In[18]:


x=df[['disp','hp','wt']]
y=df['mpg']


# In[19]:


x


# ### IMPORT THE CLASS

# In[31]:


from sklearn.linear_model import LinearRegression


# ### CREATE THE OBJECT

# In[32]:


reg = LinearRegression()


# ### TRAIN THE MODEL

# In[33]:


reg.fit(x,y)


# ### PREDICT ON UNSEEN DATA

# In[34]:


disp=221
hp=102
wt=2.91
reg.predict([[disp, hp, wt]])


# In[36]:


disp=221
hp=102
wt=3.51
reg.predict([[disp, hp, wt]])


# ### VISUALIZE 

# In[37]:


plt.figure(figsize=(16,9))
plt.subplot(2,2,1)
plt.scatter(df['disp'], y, color='r')
plt.subplot(2,2,2)
plt.scatter(df['hp'], y, color='b')
plt.subplot(2,2,3)
plt.scatter(df['wt'], y, color='g')


# In[39]:


reg.score(x,y)


# ### DECISION TREE CLASSIFICATION

# # import the dataset

# In[40]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[41]:


df.shape


# In[42]:


df


# In[43]:


list(df.columns)


# In[44]:


df


# #### Separate the input and output variables

# In[46]:


x=df[['Age','EstimatedSalary']]
y=df['Purchased']


# ### explore the data

# In[47]:


plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid()
plt.scatter(x['Age'], x['EstimatedSalary'])


# In[48]:


plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid()
plt.scatter(x['Age'], x['EstimatedSalary'], c=y)


# ### IMPORT THE CLASS

# In[50]:


from sklearn.tree import DecisionTreeClassifier


# ### create the object

# In[53]:


classifier = DecisionTreeClassifier(random_state=0)


# ### TRAIN THE MODEL

# In[55]:


classifier.fit(x,y)


# In[56]:


### Plot the tree


# In[58]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[61]:


plt.figure(figsize=(12,15))
plot_tree(classifier, fontsize=7, feature_names=['age','sal'], class_names=['y','n'], filled=True, rounded=True);


# ### prediction on unseen data

# In[63]:


new = pd.DataFrame({
    'Age':[25,37,56,66],
    'EstimateSalary': [34000,41000,26000,134000]
})


# In[64]:


new


# In[65]:


classifier.predict(new)


# In[66]:


new = pd.DataFrame({
    'Age':[25,37,56,46],
    'EstimateSalary': [340233,410030,260020,1340300]
})


# In[77]:


new = pd.DataFrame({
    'Age':[25,37,56,66],
    'EstimateSalary': [34000,41000,26000,134000]
})


# In[78]:


classifier.predict(new)


# In[79]:


### NAVIE BAYEES CLASSIICATION


# In[80]:


from sklearn.naive_bayes import GaussianNB


# In[81]:


classifier = GaussianNB()


# In[82]:


classifier.fit(x,y)


# In[83]:


classifier.predict(new)


# In[84]:


classifier.predict_proba(new)


# In[ ]:




