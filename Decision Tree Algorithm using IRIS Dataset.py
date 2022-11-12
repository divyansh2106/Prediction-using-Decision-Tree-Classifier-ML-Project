#!/usr/bin/env python
# coding: utf-8

# # DIVYANSH SINGHAL
# 
# # DATA SCIENCE INTERN - LetsGrowMore
# 
# # TASK 3 - Iris Data Prediction using Decision Tree Algorithm

# In[18]:


#Importing all the libraries required for analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[19]:


df=pd.read_csv('Iris_Dataset.csv')


# In[20]:


df.head()


# In[21]:


df.info()


# In[22]:


df.describe()


# # Building Classification Model

# In[23]:


#splitting the data
x=df[['sepal_length','sepal_width','petal_length','petal_width']]
y=df[['species']]


# In[24]:


#splitting the feature variables
x.head()


# In[8]:


#segregating the target variable
y.head()


# In[25]:


#Training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

x_train


# In[34]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[27]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()


# In[28]:


# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)


# In[29]:


#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[30]:


tree.plot_tree(clf)


# In[35]:


predictions = clf.predict(x_test)
print("The accuracy of Decision Tree is:", metrics.accuracy_score(predictions, y_test))

