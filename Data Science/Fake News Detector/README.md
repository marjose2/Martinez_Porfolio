# Building A FAke News Detector Using Python And Machine Learning
## Background:
The purpose of this project was to build a program/ model that can detect fake news articles. For the sake of this project we defince fake news as false or misleading information presented as news. 
Detecting fake news is very important society it can affect peoples ones views in politics, health, and finance. 

## The Programing

# 1. Frist, The Data Was Loaded
Code

```
#Load the data
from google.colab import files
files.upload()
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Fake%20News%20Detector/images/table1.PNG" />
</p>

This was done by gathering some fake news data. Once the data was gathered it was loeaded and took a closer look at the information.

# 2. Checked If There Was Missing Data Within This Data Set
Code

```
#Remove missing values from the data set
df.dropna(axis=0, inplace=True)
```
There was data missing, ergo the rows that had missing values were removed. 

<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Fake%20News%20Detector/images/table2.PNG" />
</p>


# 3. Combinign Columns That Were Important 
Code

```
#Show the Tokenization
df['combined'].head().apply(process_text)
```
In this step columns we combined and then tokenized the data after processing the text. 

<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Fake%20News%20Detector/images/figure1.PNG" />
</p>


## 4. created a Multinomial Naïve Bayes model and trained it on the combined data
Code
```
#Create the model
classifier = MultinomialNB()
```

## 5. See The Metrics
Code
```
from sklearn.metrics import classification_report
prep = classifier.predict(X_train)
print(Classification_report(y_train, pred))
```
<p align="center">
  <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Data%20Science/Fake%20News%20Detector/images/table3.PNG" />
</p>


Creating this model showed that about 97% accuracy on the test data (this is data that the model has nerver seem before). With alittle more dat preparation, and parameter tunning this model can be imporved and get a higher score.




