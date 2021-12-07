#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix

# Data Sourced from Kaggle competition. 
# Link: https://www.kaggle.com/saurabh00007/diabetescsv

# In[2]:


train =pd.read_csv("diabetes.csv")
train.head(5)


# Analyze Data columns

# In[3]:


train.columns


# Preform statistical analysis on features 

# In[4]:


corr = train.corr()
corr['Outcome']


# In[5]:


fig = plt.figure(figsize = (20,20))
i=1
for row in train.columns:
    fig.add_subplot(5,5,i)
    plt.hist(x = [train[train['Outcome']==1][row], train[train['Outcome']==0][row]], density=True,histtype='bar', color = ['g','r'],label = ['lower','higher'])
    plt.title(row+' by output'+str(corr['Outcome'][row]))
    plt.xlabel(row)
    plt.ylabel('# of Patients')
    plt.legend() 
    i+=1


# Define features and target variable, normalize data for easier evaluation

# In[6]:


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = train.loc[:,features ].values
y = train['Outcome']
x = MinMaxScaler().fit_transform(x)
train = pd.DataFrame(x,columns= features)
train.head(5)


# Define Global Variables for holding all fold iterations of each algorithm

# In[7]:


columns = ['TP','FP','TN','FN','TPR','TNR','FPR','FNR','Recall','Precision','F1','Accuracy','Error','BACC','TSS','HSS']
LSTMavg = pd.DataFrame(columns =columns)
RFavg = pd.DataFrame(columns =columns)
KNNavg = pd.DataFrame(columns =columns)


# Define function calculate to calculate statistical metrics using TP,TN,FP, and FN

# In[8]:


def calculate(FN,TN,FP,TP):
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    FPR = FP/(TN + FP)
    FNR = FN/(TP + FN) 
    r = TP/(TP + FN) 
    p = TP/(TP + FP)
    F1 = (2 *TP)/(2 *TP + FP + FN)
    Acc = (TP + TN)/(TP + FP + FN + TN)
    Err = (FP + FN)/(TP + FP + FN + TN)
    bacc = (TPR+TNR)/2
    TSS= TP/(TP+FN)
    HSS = 2*((TP*TN)-(FP*FN))/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    return [TP,FP,TN,FN,TPR,TNR,FPR,FNR,r,p,F1,Acc,Err,bacc,TSS,HSS]


# Define function to train, predict and analyze the Confusion Matrix

# Random Forest Algorithm

# In[9]:


def fRF(model,k,X_train, X_test, y_train, y_test):#Random Forest
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    [[TN,  FP],
    [ FN, TP]] = confusion_matrix(y_test, y_pred) 
    fold = pd.DataFrame(calculate(FN,TN,FP,TP),index = columns,columns=[k]).T
    return fold


# Long Short-Term Memory Algorithm

# In[10]:


def fLSTM(model,k,X_train, X_test, y_train, y_test):#Long Short-Term Memory
    model.fit(X_train, y_train, epochs = 10, batch_size = 32,verbose=0)
    y_pred = model.predict(X_test).reshape(-1)
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    [[TN,  FP],
    [ FN, TP]] = confusion_matrix(y_test, y_pred) 
    fold = pd.DataFrame(calculate(FN,TN,FP,TP),index = columns,columns=[k]).T
    return fold


# K-Nearest Neighbor Algorithm

# In[11]:


def fKNN(model,k,X_train, X_test, y_train, y_test):#k-Nearest Neighbor
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    [[TN,  FP],
    [ FN, TP]] = confusion_matrix(y_test, y_pred) 
    fold = pd.DataFrame(calculate(FN,TN,FP,TP),index = columns,columns=[k]).T
    return fold


# Create Random Forest Classifier

# In[12]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)


# Create LSTM Neurel Network

# In[13]:


model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (8, 1)))
model.add(Dropout(0.25))
model.add(LSTM(50))
model.add(Dropout(0.25))
model.add(Dense(units = 1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = 'accuracy')


# Create k-Nearest Neighbor Classifier

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


# Initialize k-Fold cross validation

# In[15]:


from sklearn.model_selection import KFold
X = train
kf = KFold(n_splits=10)
splits =  kf.get_n_splits(X)


# Iterate through the Kfold,train and test each algorithm at each iteration.

# In[16]:


k=1

KFold(n_splits=10, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X,y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    RFfold = fRF(clf,k,X_train, X_test, y_train, y_test)
    RFavg = pd.concat([RFavg, RFfold], ignore_index = False, axis = 0)
    LSTMfold = fLSTM(model,k,X_train, X_test, y_train, y_test) 
    LSTMavg = pd.concat([LSTMavg, LSTMfold], ignore_index = False, axis = 0)
    KNNfold = fKNN(knn,k,X_train, X_test, y_train, y_test)
    KNNavg = pd.concat([KNNavg, KNNfold], ignore_index = False, axis = 0)
    fold = pd.DataFrame([RFfold.loc[k],LSTMfold.loc[k],KNNfold.loc[k]],index = pd.Index(['Random Forest','Long Short-Term Memory','k-Nearest Neighbors'],name = "KFold k = "+str(k)))
    with pd.option_context('display.colheader_justify', 'center',
		        'display.width', 180,
                        'display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
        print(fold)
    k+=1


# Display data analysis at each k fold iteration and display average values for each algorithm

# In[17]:


avg = pd.DataFrame([RFavg.sum()/10,LSTMavg.sum()/10,KNNavg.sum()/10],index = pd.Index(['Random Forest','Long Short-Term Memory','k-Nearest Neighbors'],name = 'Overall Average Values:'))
RFavg.index.name = 'Random Forest'
LSTMavg.index.name = 'Long Short-Term Memory'
KNNavg.index.name = 'k-Nearest Neighbors'
avg.style.set_properties(**{'text-align': 'center'})
with pd.option_context('display.colheader_justify', 'center',
		    'display.width', 180,
                    'display.max_rows', None,
                    'display.max_columns', None,
                    'display.precision', 3,
                    ):
    print(RFavg)
    print(LSTMavg)
    print(KNNavg)
    print(avg)


# In[ ]:




