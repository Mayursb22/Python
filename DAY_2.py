#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


num=np.array([23,11,56,74,9,53,25])


# ### mean

# In[4]:


num.mean()


# In[5]:


np.mean(num)


# In[6]:


sum(num)/len(num)


# ### MEDIAN

# In[7]:


np.sort(num)


# In[9]:


np.median(num)


# In[10]:


x=np.arange(1,91)


# In[11]:


x


# In[12]:


np.mean(x)


# In[13]:


np.median(x)


# ### MODE

# In[15]:


import pandas as pd


# In[16]:


gender=pd.Series(['M','M','F','F','M','F','M','M','F','M'])


# In[17]:


gender.value_counts()


# In[18]:


gender.mode()


# In[23]:


age=pd.Series([22,21,20,20,21,21,20,22,23,21,20])


# In[24]:


age.value_counts()


# In[25]:


age.mode()


# ### Mid-Range

# In[26]:


(num.max() + num.min())/2


# ### Mean absolute deviarion

# In[27]:


sum(abs(num - np.mean(num)))/ len(num)


# In[28]:


x=np.ones(10)+20


# In[29]:


x


# In[32]:


sum(abs(x - np.mean(x)))/ len(x)


# ### VARIANCE

# In[33]:


np.mean((num-np.mean(num)) ** 2)


# ### STANDARD DEVIATION

# In[35]:


np.sqrt(np.mean((num - np.mean(num)) ** 2))


# In[36]:


np.std(num)


# ### DATA OPERATIONS

# In[40]:


# LOCATION : https://mitu.co.in/dataset
#Download all students* files and store in current working dir


# In[41]:





# In[43]:


df = pd.read_csv('demo.csv')


# In[44]:


df


# In[45]:


df.shape


# In[46]:


list(df.columns)


# In[47]:


df.head()


# In[48]:


df.tail()


# ### SLICING OF DATA

# In[49]:


df.iloc[3:8,2:6]


# In[51]:


df.iloc[30:,2:6]


# In[53]:


x=df.iloc[3:8,2:6]


# In[54]:


x


# In[55]:


df.iloc[3:,:6]


# In[56]:


df.iloc[:,2]


# In[57]:


df.iloc[45,:]


# In[58]:


df.loc[:,'name']


# In[59]:


df.loc[:,['name','middlename','surname']]


# In[61]:


df['native_city']


# In[62]:


df[['name','surname']]


# In[63]:


df.iloc[[2,1,45,31,28],[2,3,4,6]]


# In[66]:


df.drop([2,5,6,9])


# In[67]:


df.drop(2)


# In[68]:


df.drop('टाइमस्टँप',axis=1)


# In[70]:


df.drop('टाइमस्टँप', axis=1, inplace=True)


# In[71]:


df.columns


# In[74]:


df.dtypes


# In[75]:


df.describe()


# In[77]:


df.info()


# In[78]:


df.count()


# In[79]:


df.max()


# In[80]:


df.min()


# In[81]:


df.sum()


# In[82]:


df.mean()


# In[83]:


df.std()


# In[84]:


df.median()


# In[86]:


df['age'].median()


# In[88]:


df['age'].max()


# In[89]:


df['age'].min()


# In[90]:


df['age'].mean()


# In[91]:


df['name'].str.upper()


# In[92]:


df['name'].str.lower()


# In[93]:


df['name'].str.title()


# In[94]:


df['name'].str.strip()


# In[95]:


df['name']=df['name'].str.title()
df['middlename']=df['middlename'].str.title()
df['surname']= df['surname'].str.title()
df['native_city']= df['native_city'].str.title()
df['favorite_sport']= df['favorite_sport'].str.title()


# In[96]:


df['name']=df['name'].str.strip()
df['middlename']=df['middlename'].str.strip()
df['surname']= df['surname'].str.strip()
df['native_city']= df['native_city'].str.strip()
df['favorite_sport']= df['favorite_sport'].str.strip()


# In[97]:


df[['name','middlename','surname','native_city','favorite_sport']]


# In[98]:


df['name'].str.startswith('A')


# In[99]:


df['surname'].str.endswith('r')


# In[100]:


df['native_city'].mode()


# In[101]:


df['native_city'].value_counts()


# In[103]:


df['favorite_sport'].value_counts()


# In[104]:


df['age']>21


# In[105]:


df[df['age']>21]


# In[111]:


df[(df['age'] > 21) & (df['native_city'] == 'Pune')]


# In[110]:


df[(df['gender'] == 'Female') & (df['favorite_sport'] == 'Cricket')]['name']


# In[112]:


x=df[(df['gender']=='Female')&(df['favorite_sport']=='Cricket')]


# In[113]:


x.to_csv('output.csv',index='False')


# In[ ]:




