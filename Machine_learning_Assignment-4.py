#!/usr/bin/env python
# coding: utf-8

# In[9]:


#importing the required libraries to work with Tabular data and also to implement algorithms

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")


# # Question: 1
# 
# 
# 1. Read the provided CSV file ‘data.csv’.
# https://drive.google.com/drive/folders/1h8C3mLsso-R-sIOLsvoYwPLzy2fJ4IOF?usp=sharing
# 2. Show the basic statistical description about the data.
# 3. Check if the data has null values.
# a. Replace the null values with the mean
# 4. Select at least two columns and aggregate the data using: min, max, count, mean.
# 5. Filter the dataframe to select the rows with calories values between 500 and 1000.
# 6. Filter the dataframe to select the rows with calories values > 500 and pulse < 100.
# 7. Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”.
# 8. Delete the “Maxpulse” column from the main df dataframe
# 9. Convert the datatype of Calories column to int datatype.
# 10. Using pandas create a scatter plot for the two columns (Duration and Calories).
# 

# In[7]:


#1. Read the provided CSV file ‘data.csv’. https://drive.google.com/drive/folders/1h8C3mLsso-R-sIOLsvoYwPLzy2fJ4IOF?usp=sharing

df = pd.read_csv("data.csv")
df.head()


# In[8]:


#2. Show the basic statistical description about the data.

df.describe()


# In[10]:


#3. Check if the data has null values.

df.isnull().any()


# In[11]:


#Replace the null values with the mean

df.fillna(df.mean(), inplace=True)
df.isnull().any()


# In[12]:


#4. Select at least two columns and aggregate the data using: min, max, count, mean.

df.agg({'Maxpulse':['min','max','count','mean'],'Calories':['min','max','count','mean']})


# In[13]:


#5. Filter the dataframe to select the rows with calories values between 500 and 1000.

df.loc[(df['Calories']>500)&(df['Calories']<1000)]


# In[14]:


#6. Filter the dataframe to select the rows with calories values > 500 and pulse < 100.

df.loc[(df['Calories']>500)&(df['Pulse']<100)]


# In[15]:


#7. Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”.

df_modified = df[['Duration','Pulse','Calories']]
df_modified.head()


# In[16]:


#8. Delete the “Maxpulse” column from the main df dataframe

del df['Maxpulse']


# In[17]:


df.head()


# In[18]:


df.dtypes


# In[19]:


#9. Convert the datatype of Calories column to int datatype.

df['Calories'] = df['Calories'].astype(np.int64)
df.dtypes


# In[20]:


#10. Using pandas create a scatter plot for the two columns (Duration and Calories).

df.plot.scatter(x='Duration',y='Calories',c='blue')


# # Question: 2
# 
# Titanic Dataset
# 1. Find the correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case in class.
#    a. Do you think we should keep this feature?
# 2. Do at least two visualizations to describe or show correlations.
# 3. Implement Naïve Bayes method using scikit-learn library and report the accuracy

# In[28]:


#Loading the data file into te program
df=pd.read_csv("train.csv")

df.head()


# In[29]:


#converted categorical data to numerical values for correlation calculation

label_encoder = preprocessing.LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df.Sex.values)


#Calculation of correlation for 'Survived' and  'Sex' in data
correlation_Value= df['Survived'].corr(df['Sex'])

print(correlation_Value)


# Ans: Yes, we should keep the  'Survived' and  'Sex' features  helps classify the data accurately

# In[30]:


#print correlation matrix
matrix = df.corr()
print(matrix)


# In[31]:


# One way of visualizing correlation matrix in form of spread chart

df.corr().style.background_gradient(cmap="Reds")


# In[32]:


#Second form of visuaizing correlation matriX using heatmap() from seaborn

sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[33]:


#Loaded data files test and train and merged files

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')
train_raw['train'] = 1
test_raw['train'] = 0
df = train_raw.append(test_raw, sort=False)
features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'
df = df[features + [target] + ['train']]
df['Sex'] = df['Sex'].replace(["female", "male"], [0, 1])
df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = df.query('train == 1')
test = df.query('train == 0')


# In[34]:


# Drop missing values from the train set.
train.dropna(axis=0, inplace=True)
labels = train[target].values
train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)


# In[35]:


#Test and train split

X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)


# In[36]:


classifier = GaussianNB()

classifier.fit(X_train, Y_train)


# In[37]:


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# # Question 3
# 
# (Glass Dataset)
# 
#     1. Implement Naïve Bayes method using scikit-learn library.
#         a. Use the glass dataset available in Link also provided in your assignment.
#         b. Use train_test_split to create training and testing part.
#     2. Evaluate the model on testing part using score and classification_report(y_true, y_pred)
#     
#     1. Implement linear SVM method using scikit library
#         a. Use the glass dataset available in Link also provided in your assignment.
#         b. Use train_test_split to create training and testing part.
#     2. Evaluate the model on testing part using score and

# In[38]:


glass=pd.read_csv("glass.csv")
glass.head()


# In[39]:


glass.corr().style.background_gradient(cmap="Reds")


# In[40]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[41]:


features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'


X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass['Type'],test_size=0.2, random_state=1)

classifier = GaussianNB()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(Y_val, y_pred))


# In[43]:


from sklearn.svm import SVC, LinearSVC

classifier = LinearSVC()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# # Justification:
# We got better accuracy for Naïve Bayes method which is 0.8372093023255814. Naive Bayes analysis works well with probabilistic concepts where as Linear SVM works better with linear regression logics. But to perform more accurately SVM requires large amounts of data to train and test the data. So, due to the amount of data Naive Bayes algorith gives better accuracy compared to Linear SVM.
