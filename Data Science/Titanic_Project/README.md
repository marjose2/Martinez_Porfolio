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
<details><summary>Code</summary>

```
import numpy as np  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
</details>

###  2. Then data from seabron package was loaded and the 10 rows were printed
<details><summary>Code</summary>

```
#Load the data
titanic = sns.load_dataset('titanic')
#Print the first 10 rows of data
titanic.head(10)
```
</details>
<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/images/Ethogram.PNG" width="500" />

