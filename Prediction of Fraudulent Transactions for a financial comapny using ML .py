#!/usr/bin/env python
# coding: utf-8

# # Prediction of Fraudulent Transaction model for a Financial Company using ML

# In[ ]:


# import the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


# In[2]:


#for unwanted warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#load the dataset
df=pd.read_csv("C:\\Users\\ASUS\\Desktop\\DATA\\Fraud.csv")


# In[4]:


#show first 5 rows
df.head()


# In[5]:


#show last 5 rows
df.tail()


# In[6]:


#show 5 rows randomly
df.sample(5)


# In[7]:


#dataset informations
df.info()


# In[8]:


# show no. of rows and columns
df.shape


# In[9]:


#checking the no. of missing values
df.isnull().sum()


# # Distribution of normal transactions and fraudulent transactions

# In[10]:


print(df['isFraud'].value_counts())


# In[11]:


print(df['isFlaggedFraud'].value_counts())

# This datset is highly imbalanced.

0---> Legit Transaction
1---> Fraud  Transaction
# # Separating the data for analyis

# In[12]:


legit=df[df.isFraud==0]
fraud=df[df.isFraud==1]


# In[13]:


legit.shape


# In[14]:


fraud.shape


# In[15]:


legit=df[df.isFlaggedFraud==0]
fraud=df[df.isFlaggedFraud==1]


# In[16]:


legit.shape


# In[17]:


fraud.shape


# In[18]:


legit.describe()


# In[19]:


fraud.describe()


# # Statistical measures of the Data

# In[20]:


legit.amount.describe()


# In[21]:


fraud.amount.describe()


# # Comparing the values for both transactions

# In[22]:


df.groupby('isFraud').mean()


# In[23]:


df.groupby('isFlaggedFraud').mean()


# # Resampling

# Build a sample dataset containing similar distribution of normal & fraudulent transactions for isFraud  & isFlaggedFraud data.
# 
# No. of fraudulent transactions for isFraud data--->8213

# In[24]:


legit_downsampled = resample(legit, replace=False, n_samples=8213)


# In[25]:


fraud_upsampled = resample(fraud, replace=True , n_samples=8213)


# In[26]:


new_dataset = pd.concat([legit_downsampled , fraud_upsampled])


# In[27]:


new_dataset.isFraud.value_counts()


# No. of Fraudulent transactions for isFlaggedFraud data--- 16

# In[28]:


legit_downsampled = resample(legit, replace=False, n_samples=16)


# In[29]:


fraud_upsampled = resample(fraud, replace=True, n_samples=16)


# In[30]:


New_dataset = pd.concat([legit_downsampled, fraud_upsampled])


# In[31]:


New_dataset.isFraud.value_counts()


# In[32]:


new_dataset.groupby('isFraud').mean()


# In[33]:


New_dataset.groupby('isFlaggedFraud').mean()


# # Splitting the data into features and targets

# In[34]:


X = new_dataset.drop(columns='isFraud', axis=1)
Y = new_dataset['isFraud']


# In[35]:


X


# In[36]:


Y


# In[37]:


x = New_dataset.drop(columns='isFlaggedFraud', axis=1)
y = New_dataset['isFlaggedFraud']


# In[38]:


x


# In[39]:


x['nameOrig'].nunique()


# In[40]:


x['nameDest'].nunique()


# In[41]:


x.shape


# In[42]:


y


# In[43]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[44]:


X.shape, X_train.shape, X_test.shape


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


# In[46]:


x.shape, x_train.shape, x_test.shape


# # One Hot Encoding

# In[47]:


X_train_with_dummies = pd.get_dummies(X_train, columns = ['type', 'nameOrig', 'nameDest' ])


# In[48]:


X_train_with_dummies


# In[49]:


X_train_with_dummies.shape


# In[50]:


X_test_with_dummies = pd.get_dummies(X_test, columns = ['type', 'nameOrig', 'nameDest' ])


# In[51]:


X_test_with_dummies


# In[52]:


x_train_with_dummies = pd.get_dummies(x_train, columns = ['type', 'nameOrig', 'nameDest' ])


# In[53]:


x_train_with_dummies


# In[54]:


x_test_with_dummies = pd.get_dummies(x_test, columns = ['type', 'nameOrig', 'nameDest' ]) 


# In[55]:


x_test_with_dummies


# # Model Training

# In[56]:


#Logistic Regression


# In[57]:


model = LogisticRegression()


# In[58]:


#Training the logistic Regression model with training data.


# # Model Evaluation

# In[59]:


model.fit(X_train_with_dummies, Y_train) 


# Accuracy Score

# Accuracy on training data

# In[60]:


X_train_prediction = model.predict(X_train_with_dummies)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[61]:


print('Accuracy on Training data : ', training_data_accuracy)


# # THANK YOU
