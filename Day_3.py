#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd


# In[3]:


x=np.array([3,13,19,31,38,45,58,65,71,88])
y=np.array([11,16,24,37,41,58,56,78,81,95])
z=np.array([81,88,79,63,54,32,29,26,18,14])


# In[4]:


plt.plot(x,y)


# In[7]:


plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()
plt.xlim(0,100)
plt.ylim(0,100)
plt.plot(x,y)


# In[8]:


plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()
plt.xlim(0,100)
plt.ylim(0,100)
plt.xticks(range(0,101,10))
plt.xticks(range(0,101,10))
plt.plot(x,y)


# In[9]:


plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()
plt.xlim(0,100)
plt.ylim(0,100)
plt.xticks(range(0,101,10))
plt.xticks(range(0,101,10))
plt.plot(x,z)


# In[11]:


plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()
# markers: . o x v> ^ < D *
plt.plot(x, y, color='red', marker='o', ls=':', mfc='green')


# In[16]:


plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()

plt.plot(x, y, color='red', label= '2023')
plt.plot(x, z, color='green', label= '2022')
plt.legend()


# In[17]:


plt.subplot(2,1,1)
plt.plot(x,y)
plt.subplot(2,1,2)
plt.plot(x,z)


# In[18]:


plt.subplot(121)
plt.plot(x,y)
plt.subplot(122)
plt.plot(x,z)


# In[19]:


plt.figure(figsize=(16,9))
plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()
plt.plot(x,z, lw=5, marker='o', ms=20, mfc='y')


# In[20]:


plt.figure(figsize=(16,9))
plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()
plt.plot(x,z, lw=5, marker='o', ms=20, mfc='y')
plt.savefig('myplot.png')


# ### SCATTER PLOT

# In[21]:


plt.title('My first plot')
plt.xlabel('First value')
plt.ylabel('Second Value')
plt.grid()
plt.scatter(x,y)


# In[24]:


java=np.array([43,78,65,69,63,82,47,77,85,50,66,81])
python=np.array([46,81,62,66,69,77,41,75,88,43,65,83])
cols=[1,2,3,3,3,2,1,2,2,1,3,2]


# In[25]:


plt.title('Result')
plt.xlabel('Java Marks')
plt.ylabel('python Marks')
plt.grid()
plt.scatter(java,python)


# In[31]:


plt.title('Result')
plt.xlabel('Java Marks')
plt.ylabel('python Marks')
plt.grid()
plt.scatter(java,python,c=cols)


# In[27]:


enrollment=[104,112,256,178,126,180]
courses=['C','Java','Python','R','PHP','JavaScript']


# In[29]:


plt.xlabel('Courses')
plt.ylabel('Enrolled students')
plt.grid()
plt.bar(courses,enrollment,color='brown')


# In[30]:


plt.xlabel('Courses')
plt.ylabel('Enrolled students')
plt.grid()
plt.barh(courses,enrollment,color='brown')


# ### PIE CHART

# In[36]:


plt.figure(figsize=(10, 10))
plt.pie(enrollment, labels=courses, autopct='%2.2f', shadow=True);


# ### UNIVARIATE ANALYSIS

# ##HISTOGRAM

# In[37]:


marks = np.array([67,56,45,61,93,72,65,58,65,49,68,62])


# In[39]:


plt.title('Frequency plot of Marks')
plt.xlabel('Range of marks')
plt.ylabel('count')
plt.grid()
plt.xticks(range(0,101,10))
plt.hist(marks, bins=range(0,101,10), color='orange');


# ### BOX PLOT

# In[40]:


plt.boxplot(marks);


# ### MACHINE LEARNING 

# ### REGRESSION

# In[42]:


#dataset: salary_data.csv, Social_Newtwork_Ads.csv Mall_Customers.csv
#location: https://mitu.co.in/dataset


# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[50]:


df=pd.read_csv('Salary_Data.csv')


# In[51]:


df.shape


# In[52]:


import os
os.getcwd()


# In[53]:


df


# In[56]:


plt.scatter(df['YearsExperience'],df['Salary'])


# In[59]:


x=df['YearsExperience'].values
y=df['Salary']


# In[67]:


x


# In[84]:


y;


# In[85]:


x=x.reshape(30, 1)


# In[86]:


x


# In[87]:


from sklearn.linear_model import LinearRegression


# ### create the object of linear regression

# In[88]:


regressor=LinearRegression()


# ### train the alogrithm

# In[89]:


regressor.fit(x, y)


# In[90]:


regressor.predict([[20]])


# In[83]:


regressor.score(x, y)


# In[75]:


regressor.predict([[15]])


# In[ ]:




