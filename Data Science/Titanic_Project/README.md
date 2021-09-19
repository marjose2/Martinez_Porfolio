# Titanic Project Background 
The intedn of this project was to make two prediction from the data set given. The frist prediction being which passengers on board the Titanic would survive; the second prediction is to see if we would’ve survived.


## Data Set Description
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

<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Titanic_Project/images/figure 1.PNG"/>

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




















































## Contact Me:

+ Profesional Email: joseignacio1225@hotmail.com
+ Linkedin: https://www.linkedin.com/in/jose-martinez-10303b1aa/

