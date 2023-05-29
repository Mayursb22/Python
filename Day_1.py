#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("hello world")


# ### ARRAY OPERATIONS

# In[5]:


import numpy as np


# In[6]:


num = np.array([56,23,11,67,33,24,213,11])
print(num)


# In[7]:


type(num)


# In[11]:


num
num.dtype
num.size
num[3]
num[3]=35
num
num=np.array([56,23,11,67,42,38,24,64])
num
num[1]
num=np.array([56,23,11,67,42,38,24,True])
num
num=np.array([56,23,11,67,'hello',38,24,64])
num
city=np.array(["तुज्या वर प्रेम करतो"])


# In[12]:


num.dtype


# In[13]:


num.size
num[3]
num[3]=35
num
num=np.array([56,23,11,67,42,38,24,64])
num
num[1]
num=np.array([56,23,11,67,42,38,24,True])
num
C
num


# In[17]:


city=np.array(["तुज्या वर प्रेम करतो "])


# In[18]:


city


# In[19]:


city[0]*5


#  num.max()
#  

# In[20]:


num.max()


# In[21]:


num=np.array([56,23,11,67,42,38,24,True])
num.std()
num.max()


# In[22]:


num.std()


# In[23]:


help(num)


# In[24]:


x=[56,22,33]
x*2


# In[25]:


num*3


# In[26]:


num/10


# In[27]:


num>40


# In[30]:


num-10


# num - 10

# In[31]:


num - 10


# In[32]:


num=np.array([56,23,11,67,42,38,24,True])
num-10


# In[33]:


42 in num


# In[34]:


num = 10


# In[35]:


num


# In[36]:


num.sort()


# In[37]:


list(range(1,11))


# In[38]:


list(range(1,11,-1))


# In[39]:


list(range(11,1,-1))


# In[40]:


np.arange(20,0,-2)


# In[41]:


np.arange(1,10,0.25)


# In[43]:


np.empty(10)
np.zeros(12)
np.linspace(1,10,20)


# In[47]:


x=np.array([[4,5,6],[1,8,2]])
x


# In[50]:


x.shape
x.size


# In[51]:


x.shape


# In[52]:


x[1][2]


# In[53]:


x.reshape(10,10)


# In[54]:


x.reshape(6,1)


# In[55]:


x.reshape(-1,2)


# In[56]:


x.flatten()


# ##### SERIES OPERATIONS 

# In[65]:


import pandas as pd


# In[80]:


s=pd.Series([20,22,223,32])


# In[83]:


s=pd.Series([20,22,223,32], index=range(100,104))


# In[70]:


s.dtype


# In[84]:


s.size


# In[73]:


s


# In[74]:


s.dtype


# In[79]:


s[104]


# In[85]:


x=pd.Series([1,2,3,4,5])
y=pd.Series(['mayur','ram','dash','rash','prash'])
z=pd.Series([67.99,53.11,31.33,31.33,33.44])


# In[87]:


df=pd.DataFrame({
    'roll':x,
    'name':y,
    'marks':z
})


# In[88]:


df


# In[89]:


df.shape


# In[90]:


df.columns


# In[91]:


df.size


# In[92]:


df.T


# In[93]:


list(s)


# In[94]:


dict(df)


# In[96]:


df.values


# In[103]:


x=np.arange(1,100)
x


# In[105]:


y=np.array([67,12,33,97,75])


# In[107]:


y.mean()


# In[109]:


x.mean()


# In[ ]:




