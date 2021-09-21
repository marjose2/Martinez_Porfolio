# 🌊Titanic Project Background🛳️
The intedn of this project was to make two prediction from the data set given. The frist prediction being which passengers on board the Titanic would survive; the second prediction is to see if we would’ve survived.


## DataSet Description
|Variable               | Defenition                | Key              |
|-----------------------|----------------------------|------------------|
|Survival| Survival| 0 = no, 1 = yes|
|pclass| Ticket class| 1 = 1st, 2 = 2nd, 3 = 3rd|
|sex| sex| |
|age| age in years| |
|sibsp| # of siblings/ spouses aboard the titanic| |
|parch| # of parent / children aboard the Titanic| |
|ticket| Ticket number||
|fare| Passenger fare||
|cabin| cabin number||
|embarked| port of embarktion| C = Cherbourg, Q = Queenstown, S = Southampton|

<details><summary> More details about the data set / variables </summary>

pclass: A proxy for socio-economic status (SES); 1st = Upper, 2nd = Middle, 3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
; Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
; Child = daughter, son, stepdaughter, stepson
**Some children travelled only with a nanny, therefore parch=0 for them.
</details>

##Process: 
###  1. This project started by importing the packeges/ libraries to me make it easier to write the program
```
import numpy as np  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

###  2. Then data from seabron package was loaded and the 10 rows were printed

```
#Load the data
titanic = sns.load_dataset('titanic')
#Print the first 10 rows of data
titanic.head(10)
```
<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/table1.PNG" width="1000" />

### 3. Analyzed the data by:
#### a. Counting the number of rows in a Data Set 

```
#Count the number of rows and columns in the data set 
titanic.shape
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure 1.PNG"/>
</p>


#### b. Got some statictics


```
#Get some statistics
titanic. describe()
```

<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/table2.PNG" width="1000" />

#### c. Got a count of the number of survivors


```
#Get a count of the number of survivor
titanic['survived'].value_counts()
```


### 4. Made a visualization of the data
#### a. Vizualize the count of survivors for the columns who, se, pclass, sibsp, parch, and embarked


```
cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']

n_rows = 2
n_cols = 3

# The subplot grid and the figure size of each graph
# This returns a Figure (fig) and an Axes Object (axs)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):  
        
        i = r*n_cols+ c #index to go through the number of columns       
        ax = axs[r][c] #Show where to position each subplot
        sns.countplot(titanic[cols[i]], hue=titanic["survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right') 
        
plt.tight_layout()
```
  
<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/firgure 3.PNG" width="1000" />
This figure allowed one to see that that a male, 18 or older, is not likely to survive from the chart ``who``. Females are most likely to survive from the chart ```sex```. Third class is most likely to not survive by chart ```pclass```. If you have 0 siblings or spouses on board, you are not likely to survive according to chart ```sibsp```. If you have 0 parents or children on board, you are not likely to survive according to the ```parch``` chart. If you embarked from Southampton (S), you are not likely to survive according to the ```embarked``` chart.


### 5. Then closer look at at survival rate by sex
#### a. Male vs Female Survival Rate Table 
Code
```
#Look at survival rate by sex
titanic.groupby('sex')[['survived']].mean()
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/table3.PNG" />
</p>

This table shows that about 74.2% of females survived and about 18.89% of males survived.

#### b. Survival Rate by Sex and Class Table
Code

```
#Look at survival rate by sex and class
titanic.pivot_table('survived', index='sex', columns='class')
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/table4.PNG" />
</p>
This table shows that that females in first class had a survival rate of about 96.8%, meaning the majority of them survived, While males in third class had the lowest survival rate at about 13.54%, meaning the majority of them did not survive.


#### c. Visualizing step 5.b (Survival Rate by Sex and Class)
Code

```
#Look at survival rate by sex and class visually
titanic.pivot_table('survived', index='sex', columns='class').plot()
```

<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure4.PNG" />
</p>



#### d. Visualizing step 5.b (Survival Rate by Sex and Class) Using a Bar Plot
Code
```
#Plot the survival rate of each class.
sns.barplot(x='class', y='survived', data=titanic)
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure5.PNG" />
</p>

In this bar plot we can see that over 60% of the passengers in first class survived. Moreover, less than 30% of passengers in third class survived. Meaning that means less than half of the passengers in third class survived, compared to the passengers in first class.

### 6. Look at Survival Rate by Sex, Age, and Class
Code

```
#Look at survival rate by sex, age and class
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')
```

<p align="center">
    <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/table5.PNG" />
    </p>
    
Here we can see that the oldest person is aged 80. Women in first class that were 18 and older had the highest survival rate at 97.2973%. Men 18 and older in second class had the lowest survival rate of 7.1429%.

### 7. We Ploted the Prices for Each Class
Code 
```
#Plot the Prices Paid Of Each Class
  plt.scatter(titanic['fare'], titanic['class'],  color = 'purple', label='Passenger Paid')
  plt.ylabel('Class')
  plt.xlabel('Price / Fare')
  plt.title('Price Of Each Class')
  plt.legend()
  plt.show()
```

<p align="center">
   <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure6.PNG" />
    </p>


### 8. Look at the Cloumns Contaning Empty Values 
Code 

```
#Count the empty values in each column 
titanic.isna().sum()
```

Looks like the columns containing empty values are age, embarked, deck, and embarked_town.


<p align="center">
   <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure7.PNG" />
    </p>
    

### 9. See wich Columns are Redundant
Code

```
#Look at all of the values in each column & get a count 
for val in titanic:
   print(titanic[val].value_counts())
   print()
```
#### a. Drop the Columns that are Missing Values and that are Redundant
 Code
```
 # Drop the columns
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)

#Remove the rows with missing values
titanic = titanic.dropna(subset =['embarked', 'age'])
```

#### b. See the new number of rows and columns in the data set
Code

```
#Count the NEW number of rows and columns in the data set
titanic.shape
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure8.PNG"/>
</p>


### 10. Look at Data tyoes that need to be transformed
Code

```
titanic.dtypes
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure9.PNG"/>
</p>


### 11. Print the unique values of the non-numeric data
Code

```
#Print the unique values in the columns
print(titanic['sex'].unique())
print(titanic['embarked'].unique())
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure10.PNG"/>
</p>


### 12. Changed the non-numeric data, and print the new values
Code

```
#Encoding categorical data values (Transforming object data types to integers)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encode sex column
titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)
#print(labelencoder.fit_transform(titanic.iloc[:,2].values))

#Encode embarked
titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)
#print(labelencoder.fit_transform(titanic.iloc[:,7].values))

#Print the NEW unique values in the columns
print(titanic['sex'].unique())
print(titanic['embarked'].unique())
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure11.PNG"/>
</p>


### 13. Split the dat into 'X' and dependent 'Y' data sets
Code

```
#Split the data into independent 'X' and dependent 'Y' variables
X = titanic.iloc[:, 1:8].values 
Y = titanic.iloc[:, 0].values 
```
### 14. Split the data again, this time into 80% training (X_train and Y_train) and 20% testing (X_test and Y_test) data sets
Code

```
# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```

### 15. Scale the data (the dat will be within a specific range)
Code

```
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### 16. Create a funtion that has within it many different machine learning models that we can use to make our predictions
Code

```
#Create a function within many Machine Learning Models
def models(X_train,Y_train):
  
  #Using Logistic Regression Algorithm to the Training Set
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest
```

### 17. Get atrain models and store them in variable called model
Code

```
#Get and train all of the models
model = models(X_train,Y_train)
```

<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure12.PNG"/>
</p>


### 18. Show  the confusion matrix and accuracy for all the models on the test data
Code

```

from sklearn.metrics import confusion_matrix 
for i in range(len(model)):
   cm = confusion_matrix(Y_test, model[i].predict(X_test)) 
   #extracting TN, FP, FN, TP
   TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
    print(cm)
   print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
   print()# Print a new line
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure13.PNG"/>
</p>


-  False Positive (FP)= A test result which incorrectly indicates that a particular condition or attribute is present.
-  True Positive (TP)= Sensitivity (also called the true positive rate, or probability of detection in some fields), measures the proportion of actual positives that are -  correctly identified as such.
-  True Negative (TN)= Specificity (also called the true negative rate), measures the proportion of actual negatives that are correctly identified as such.
-  False Negative (FN)= A test result that indicates that a condition does not hold, while in fact, it does. For example, a test result that indicates a person does not survive when the person actually does.

### 19. Get the important features
Code

```
#Get the importance of the features
forest = model[6]
importances = pd.DataFrame({'feature':titanic.iloc[:, 1:8].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances
```

<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure14.PNG"/>
</p>


### 20. Visualize the important features
Code

```
#Visualize the importance
importances.plot.bar()
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure15.PNG"/>
</p>


### 21. Print the Random Forest Classifier Model predictions for each passenger and, below it, print the actual values
Code

```

#Print Prediction of Random Forest Classifier model
pred = model[6].predict(X_test)
print(pred)

#Print a space
print()

#Print the actual values
print(Y_test)
```

<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure16.PNG"/>
</p>


### 22. Creating a variable called my survival
Code

```
my_survival = [[3,1,21,0, 0, 0, 1]]
#Print Prediction of Random Forest Classifier model
pred = model[6].predict(my_survival)
print(pred)

if pred == 0:
  print('Oh no! You didn't make it')
else:
  print('Nice! You survived')
```

-  In it, I will have a pclass = 3, meaning I would probably be in the third class because of the cheaper price.
-  I am a male, so sex = 1.
-  I am older than 18, so I will put age = 22.
-  I would not be on the ship with siblings or spouses, so sibsp = 0.
-  No children or parents, so parch = 0.
-  I would try to pay the minimum fare, so fare = 0.
-  I would’ve embarked from Queenstown, so embarked = 1.

<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure17.PNG"/>
</p>



## Contact Me:

+ Profesional Email: joseignacio1225@hotmail.com
+ Linkedin: https://www.linkedin.com/in/jose-martinez-10303b1aa/

