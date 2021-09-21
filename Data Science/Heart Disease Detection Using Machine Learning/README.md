# Heart Attack Analysis And Prediction
## Background:
## Process:
### 1. Importing all the libaries required 
Code

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
### 2. Import the local machine
Code

```
data1 = pd.read_csv('/kaggle/input/heart-attack-analysis-prediction-dataset/heart.csv')
data1 = pd.DataFrame(data1)
```

### 3. Look quick at the data
Code

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trtbps    303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalachh  303 non-null    int64  
 8   exng      303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slp       303 non-null    int64  
 11  caa       303 non-null    int64  
 12  thall     303 non-null    int64  
 13  output    303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
```

### 4. Look at the number of record and fetures in the dataset
Code

```
#number of records and features in the dataset
data1.shape
```

### 5. Look for dupicate data
Code

```
#Check duplicate rows in data
duplicate_rows = data1[data1.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape)
```

### 6. Remove the duplicate rows
Code
```
#we have one duplicate row.
#Removing the duplicate row

data1 = data1.drop_duplicates()
duplicate_rows = data1[data1.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape)

#Number of duplicate rows after dropping one duplicate row
```

### 7. Check to see if if the data is consistent
Code

```
#Check if the other data is consistent
data1.shape
```

### 8. Look for null values
Code

```
#Looking for null values
print("Null values :: ")
print(data1.isnull() .sum())
```

### 9. Look for outliars
Code

```
# 1. Detecting Outliers using IQR (InterQuartile Range)
sns.boxplot(x=data1['age'])
```
Repeat this process with other variables (sex, cp, trtps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall,  )

### 10. Find the interquarule Range
Code 

```
#Find the InterQuartile Range
Q1 = data1.quantile(0.25)
Q3 = data1.quantile(0.75)

IQR = Q3-Q1
print('*********** InterQuartile Range ***********')
print(IQR)
```

### 11. Removing the outliars 
#### a. using IQR
Code 

```
# Remove the outliers using IQR
data2 = data1[~((data1<(Q1-1.5*IQR))|(data1>(Q3+1.5*IQR))).any(axis=1)]
data2.shape
```

#### b. Using Z-score
Code

```
#Removing outliers using Z-score
z = np.abs(stats.zscore(data1))
data3 = data1[(z<3).all(axis=1)]
data3.shape
```
### 12. Fiding the correlation between variables
Code

```

#Finding the correlation between variables
pearsonCorr = data3.corr(method='pearson')
spearmanCorr = data3.corr(method='spearman')
```


### 13. Visual representtion of the correlation
#### a. Pearson Correlation
Code

```
fig = plt.subplots(figsize=(14,8))
sns.heatmap(pearsonCorr, vmin=-1,vmax=1, cmap = "Greens", annot=True, linewidth=0.1)
plt.title("Pearson Correlation")
```
#### b. Spearman Correlation
Code

```
fig = plt.subplots(figsize=(14,8))
sns.heatmap(spearmanCorr, vmin=-1,vmax=1, cmap = "Blues", annot=True, linewidth=0.1)
plt.title("Spearman Correlation")
```

#### c. Creating a mask for both correlation matrices, Pearson Corr mask
Code

```
#Create mask for both correlation matrices

#Pearson corr masking
#Generating mask for upper triangle
maskP = np.triu(np.ones_like(pearsonCorr,dtype=bool))

#Adjust mask and correlation
maskP = maskP[1:,:-1]
pCorr = pearsonCorr.iloc[1:,:-1].copy()

#Setting up a diverging palette
cmap = sns.diverging_palette(0, 200, 150, 50, as_cmap=True)

fig = plt.subplots(figsize=(14,8))
sns.heatmap(pCorr, vmin=-1,vmax=1, cmap = cmap, annot=True, linewidth=0.3, mask=maskP)
plt.title("Pearson Correlation")
```


#### d. Creating a mask for both correlation matrices, Spearson Corr mask
Code

```
#Create mask for both correlation matrices

#Spearson corr masking
#Generating mask for upper triangle
maskS = np.triu(np.ones_like(spearmanCorr,dtype=bool))

#Adjust mask and correlation
maskS = maskS[1:,:-1]
sCorr = spearmanCorr.iloc[1:,:-1].copy()

#Setting up a diverging palette
cmap = sns.diverging_palette(0, 250, 150, 50, as_cmap=True)

fig = plt.subplots(figsize=(14,8))
sns.heatmap(sCorr, vmin=-1,vmax=1, cmap = cmap, annot=True, linewidth=0.3, mask=maskS)
plt.title("Spearman Correlation")
```

### 14. Buiding a classification model
Code

```

x = data3.drop("output", axis=1)
y = data3["output"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
````
### 15. Building classification models
Code

```
names = ['Age', 'Sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']

#Logistic Regression
logReg = LogisticRegression(random_state=0, solver='liblinear')
logReg.fit(x_train, y_train)

#Check accuracy of Logistic Regression
y_pred_logReg = logReg.predict(x_test)
#Model Accuracy
print("Accuracy of logistic regression classifier :: " ,metrics.accuracy_score(y_test,y_pred_logReg))

#Removing the features with low correlation and checking effect on accuracy of model
x_train1 = x_train.drop("fbs",axis=1)
x_train1 = x_train1.drop("trtbps", axis=1)
x_train1 = x_train1.drop("chol", axis=1)
x_train1 = x_train1.drop("restecg", axis=1)

x_test1 = x_test.drop("fbs", axis=1)
x_test1 = x_test1.drop("trtbps", axis=1)
x_test1 = x_test1.drop("chol", axis=1)
x_test1 = x_test1.drop("restecg", axis=1)

logReg1 = LogisticRegression(random_state=0, solver='liblinear').fit(x_train1,y_train)
y_pred_logReg1 = logReg1.predict(x_test1)
print("\nAccuracy of logistic regression classifier after removing features:: " ,metrics.accuracy_score(y_test,y_pred_logReg1))
```

### 16. Maku=ing the Decision tree Classification
Code

```

decTree = DecisionTreeClassifier(max_depth=6, random_state=0)
decTree.fit(x_train,y_train)

y_pred_decTree = decTree.predict(x_test)

print("Accuracy of Decision Trees :: " , metrics.accuracy_score(y_test,y_pred_decTree))

#Remove features which have low correlation with output (fbs, trtbps, chol)
x_train_dt = x_train.drop("fbs",axis=1)
x_train_dt = x_train_dt.drop("trtbps", axis=1)
x_train_dt = x_train_dt.drop("chol", axis=1)
x_train_dt = x_train_dt.drop("age", axis=1)
x_train_dt = x_train_dt.drop("sex", axis=1)

x_test_dt = x_test.drop("fbs", axis=1)
x_test_dt = x_test_dt.drop("trtbps", axis=1)
x_test_dt = x_test_dt.drop("chol", axis=1)
x_test_dt = x_test_dt.drop("age", axis=1)
x_test_dt = x_test_dt.drop("sex", axis=1)

decTree1 = DecisionTreeClassifier(max_depth=6, random_state=0)
decTree1.fit(x_train_dt, y_train)
y_pred_dt1 = decTree1.predict(x_test_dt)

print("Accuracy of decision Tree after removing features:: ", metrics.accuracy_score(y_test,y_pred_dt1))
```

### 17. Random Forest Classifier
Code

```

# Using Random forest classifier

rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)

print("Accuracy of Random Forest Classifier :: ", metrics.accuracy_score(y_test, y_pred_rf))

#Find the score of each feature in model and drop the features with low scores
f_imp = rf.feature_importances_
for i,v in enumerate(f_imp):
    print('Feature: %s, Score: %.5f' % (names[i],v))
```

### 18. Removing : fbs(score=0.006), sex(score=0.02), trtbps(score=0.072), chol(score=0.078), restecg(score=0.02), exng(score=0.06), slp(score=0.06)
Code

```
names1 = ['Age', 'Sex', 'cp', 'trtbps', 'chol','restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']

x_train_rf2 = x_train.drop("fbs",axis=1)
#x_train_rf2 = x_train_rf2.drop("sex",axis=1)
#x_train_rf2 = x_train_rf2.drop("restecg",axis=1)
#x_train_rf2 = x_train_rf2.drop("slp",axis=1)
#x_train_rf2 = x_train_rf2.drop("exng",axis=1)
#x_train_rf2 = x_train_rf2.drop("trtbps",axis=1)
#x_train_rf2 = x_train_rf2.drop("chol",axis=1)
#x_train_rf2 = x_train_rf2.drop("age",axis=1)

x_test_rf2 = x_test.drop("fbs", axis=1)
#x_test_rf2 = x_test_rf2.drop("sex", axis=1)
#x_test_rf2 = x_test_rf2.drop("restecg",axis=1)
#x_test_rf2 = x_test_rf2.drop("slp",axis=1)
#x_test_rf2 = x_test_rf2.drop("exng",axis=1)
#x_test_rf2 = x_test_rf2.drop("trtbps",axis=1)
#x_test_rf2 = x_test_rf2.drop("chol",axis=1)
#x_test_rf2 = x_test_rf2.drop("age",axis=1)

rf2 = RandomForestClassifier(n_estimators=500)
rf2.fit(x_train_rf2,y_train)

y_pred_rf2 = rf2.predict(x_test_rf2)
print("Accuracy of Random Forest Classifier after removing features with low score :")
print("New Accuracy :: " , metrics.accuracy_score(y_test,y_pred_rf2))
print("\n")
print("---------------------------------------------------------------------------------------------")

f_imp = rf2.feature_importances_
for i,v in enumerate(f_imp):
    print('Feature: %s, Score: %.5f' % (names1[i],v))
print("---------------------------------------------------------------------------------------------")
````

### 19. K neighbours Classifier
Code

```
#K Neighbours Classifier

knc =  KNeighborsClassifier()
knc.fit(x_train,y_train)

y_pred_knc = knc.predict(x_test)

print("Accuracy of K-Neighbours classifier :: ", metrics.accuracy_score(y_test,y_pred_knc))
```

### 20. Testing the Models Acurracy
Code

```
#Models and their accuracy

print("*****************Models and their accuracy*****************")
print("Logistic Regression Classifier :: ", metrics.accuracy_score(y_test,y_pred_logReg1))
print("Decision Tree :: ", metrics.accuracy_score(y_test,y_pred_dt1))
print("Random Forest Classifier :: ", metrics.accuracy_score(y_test, y_pred_rf))
print("K Neighbours Classifier :: ", metrics.accuracy_score(y_test,y_pred_knc))
```














