
The Data
After taking a look at the data, there are 137 samples in the training set and 100,000 samples in the test set. This is very intriguing since the distribution of data is usually the other way around. The goal here would be to model revenue based on 137 samples in the training set and see how well the model performs on the 100,000 samples in the test set. The data fields for each sample consist of the restaurant ID which is unique for each restaurant in the sample, the opening date of the restaurant, the city, city group, restaurant type, several non-arbitrary P-variables, and revenue which is the target variable. Using a complex model for this small training dataset with noise will cause the model to overfit to the dataset. To prevent that from happening, regularization techniques for linear regression will definitely need to be used.

Null values
image 1
After a brief look at the training data, it appears that there are no null values which is a good thing. However, that may not be the case for the P-variables as we will see later in the data exploration.

Data Pre-Processinf and Exploring Fetures 
Type
image 2
image 3
The two figures above show the count of types of restaurants in the training set and test set. Looking carefully, there doesn’t seem to be a single occurrence of the ‘MB’ type in the training set. Type ‘MB’ stands for mobile restaurants and type ‘DT’ stands for drive-thru restaurants. Since mobile restaurants are more related to drive-thru than inline and food courts, the ‘MB’ samples in the test set were replaced with the ‘DT’ type.
image 3
image 4
There doesn’t seem to be any changes required for the city group feature. The training set has slightly more ‘Big Cities’ samples than ‘Other’ samples but that shouldn’t be a problem when we create our model. It should also be intuitive that restaurant revenue in the city than other areas.
City
```
(df['City'].nunique(), test_df['City'].nunique())
Out[10]: (34, 57)
```
For the ‘City’ feature, it appears that there are cities in the test set that aren’t in the training set. It is also worth noting that some of the non-arbitrary P-variables already contain geolocation information so the entire ‘City’ feature was dropped for both datasets.

Open Date
```
import datetime
df.drop('Id',axis=1,inplace=True)
df['Open Date']  = pd.to_datetime(df['Open Date'])
test_df['Open Date']  = pd.to_datetime(test_df['Open Date'])
launch_date = datetime.datetime(2015, 3, 23)
# scale days open
df['Days Open'] = (launch_date - df['Open Date']).dt.days / 1000
test_df['Days Open'] = (launch_date - test_df['Open Date']).dt.days / 1000
df.drop('Open Date', axis=1, inplace=True)
test_df.drop('Open Date', axis=1, inplace=True)
```
The opening date is the date the restaurant first opened. It won’t be of much use in terms of predicting revenue but it would be useful to know how long the restaurant has been open since the opening date. For that reason, I decided to use March 23, 2015 as the date of comparison to calculate the amount of days the restaurant has been open. Then, I chose to downscale the number of days open by a factor of 1000 to slightly improve model performance.

P-Varables
The data has 37 p-variables which are all obfuscated data. These features contain demographic data, real estate data, and commercial data based on the data field description on the Kaggle competition page.
image 5

Initially, I had thought that the p-variables were numerical features but after reading some of the discussions in the competition, it turns out that some of these features were actually categorical data encoded using integers. What’s even more interesting is that a majority of the values for some of these features are zero. Once again, after digging through the discussions, people concluded that these zero values were actually null values as shown in the plots above. Multivariate imputation by chained equations (also known as MICE) was used to replace the missing values in some of these features. The way it works is that is uses the entire set of available data to estimate the missing values.

```
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp_train = IterativeImputer(max_iter=30, missing_values=0, sample_posterior=True, min_value=1, random_state=37)
imp_test = IterativeImputer(max_iter=30, missing_values=0, sample_posterior=True, min_value=1, random_state=23)

p_data = ['P'+str(i) for i in range(1,38)]
df[p_data] = np.round(imp_train.fit_transform(df[p_data]))
test_df[p_data] = np.round(imp_test.fit_transform(test_df[p_data]))
```

The imputer was used on all p-variables separately for the training set and the test set. The missing values are estimated several times before the imputer takes the average. Before feeding these averages to the model, they need to be rounded to the nearest integer.


Once Hot Encoding
To deal with object types in the data, one hot encoding will be used to transform these features into numerical form which can be provided to the machine learning models. Dummy encoding can also be used to avoid redundancy. The features that will be encoded are ‘Type’ and ‘City Group’ since they are the only object types in the datasets.

```
columnsToEncode = df.select_dtypes(include=[object]).columns
df = pd.get_dummies(df, columns=columnsToEncode, drop_first=False)
test_df = pd.get_dummies(test_df, columns=columnsToEncode, drop_first=False)
```

Target Variable Distribution
image 6
Based on the distribution, it looks like revenue is right skewed. There also appears to be outliers which will cause issues in model training. Since we will be experimenting with linear models, the target variable will be transformed to make it normally distributed for improved model interpretation. The target variable was log transformed so the final predictions will need to be exponentiated to rescale the results back to normal.
image 7

Model Experimentation
The models that I decided on experimenting with were several different linear models, KNN, random forest and gradient boosted models. The goal here is to find the best hyper-tuned models to ensemble for the final model. Before we train any model, we will split the training set into a training and validation set.
```
df['revenue'] = np.log1p(df['revenue'])
X, y = df.drop('revenue', axis=1), df['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=118)
```
Ridge Linear Model
Ridge regression is a regularized linear model. As stated earlier, regularization techniques need to be used to prevent overfitting especially since our training set is very small. Before we train a ridge model on the training, we need to find the optimal parameters for the model. To do this, grid search along with k-fold cross validation was used to find the optimal parameters that led to the best score.
```
params_ridge = {
    'alpha' : [.01, .1, .5, .7, .9, .95, .99, 1, 5, 10, 20],
    'fit_intercept' : [True, False],
    'normalize' : [True,False],
    'solver' : ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

ridge_model = Ridge()
ridge_regressor = GridSearchCV(ridge_model, params_ridge, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
ridge_regressor.fit(X_train, y_train)
print(f'Optimal alpha: {ridge_regressor.best_params_["alpha"]:.2f}')
print(f'Optimal fit_intercept: {ridge_regressor.best_params_["fit_intercept"]}')
print(f'Optimal normalize: {ridge_regressor.best_params_["normalize"]}')
print(f'Optimal solver: {ridge_regressor.best_params_["solver"]}')
print(f'Best score: {ridge_regressor.best_score_}')
```
result:
Optimal alpha: 1.00
Optimal fit_intercept: True
Optimal normalize: True
Optimal solver: saga
Best score: -0.4463902820636504

The optimal parameters are then used for model evaluation using both the training and test sets. The RMSE here is actually RMSLE since we’ve taken the log of the target variable.

```
ridge_model = Ridge(alpha=ridge_regressor.best_params_["alpha"], fit_intercept=ridge_regressor.best_params_["fit_intercept"], 
                    normalize=ridge_regressor.best_params_["normalize"], solver=ridge_regressor.best_params_["solver"])
ridge_model.fit(X_train, y_train)
y_train_pred = ridge_model.predict(X_train)
y_pred = ridge_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```

result:
Train r2 score:  -8.194131557202557
Test r2 score:  0.04071513024835127
Train RMSE: 0.4031
Test RMSE: 0.5413

```
# Ridge Model Feature Importance
ridge_feature_coef = pd.Series(index = X_train.columns, data = np.abs(ridge_model.coef_))
ridge_feature_coef.sort_values().plot(kind = 'bar', figsize = (13,5));
```
image 8
Lasso Linear Model
Now we repeat the same procedure for a lasso model. The lasso model works differently from the ridge model because it shrinks the coefficients of less important features. This can be visualized later in the feature importance plot.

```
lasso_model = Lasso(alpha=lasso_regressor.best_params_["alpha"], fit_intercept=lasso_regressor.best_params_["fit_intercept"], 
                    normalize=lasso_regressor.best_params_["normalize"])
lasso_model.fit(X_train, y_train)
y_train_pred = lasso_model.predict(X_train)
y_pred = lasso_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```

result:
Train r2 score:  -327.6628679746241
Test r2 score:  0.02293335570713051
Train RMSE: 0.4508
Test RMSE: 0.5463

```
lasso_model = Lasso(alpha=lasso_regressor.best_params_["alpha"], fit_intercept=lasso_regressor.best_params_["fit_intercept"], 
                    normalize=lasso_regressor.best_params_["normalize"])
lasso_model.fit(X_train, y_train)
y_train_pred = lasso_model.predict(X_train)
y_pred = lasso_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```

result:
Train r2 score:  -327.6628679746241
Test r2 score:  0.02293335570713051
Train RMSE: 0.4508
Test RMSE: 0.5463

```
# Lasso Model Feature Importance
lasso_feature_coef = pd.Series(index = X_train.columns, data = np.abs(lasso_model.coef_))
lasso_feature_coef.sort_values().plot(kind = 'bar', figsize = (13,5));
```

We can see that the lasso model is generalizing a lot better than the ridge model using just the ‘Days Open’ feature. It’s able to achieve just about the same test error as the ridge model using all features which shows the true potential of these regularization techniques.
ElasticNet
ElasticNet is a linear model that combines the regularization techniques of ridge and lasso. We will use ElasticNetCV to select the best hybrid model using cross validation.

```
from sklearn.linear_model import ElasticNetCV, ElasticNet

# Use ElasticNetCV to tune alpha automatically instead of redundantly using ElasticNet and GridSearchCV
el_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=5e-2, cv=10, n_jobs=-1)         
el_model.fit(X_train, y_train)
print(f'Optimal alpha: {el_model.alpha_:.6f}')
print(f'Optimal l1_ratio: {el_model.l1_ratio_:.3f}')
print(f'Number of iterations {el_model.n_iter_}')
```
result:
Optimal alpha: 0.622309
Optimal l1_ratio: 0.100
Number of iterations 34

```
y_train_pred = el_model.predict(X_train)
y_pred = el_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```
result:
Train r2 score:  -9.795109289625826
Test r2 score:  0.05585927421195669
Train RMSE: 0.4128
Test RMSE: 0.5370
There is little to no improvement using the elastic net model. We can see that the training scores and test scores between the linear models are about the same.

```
# ElasticNet Model Feature Importance
el_feature_coef = pd.Series(index = X_train.columns, data = np.abs(el_model.coef_))
n_features = (el_feature_coef>0).sum()
print(f'{n_features} features with reduction of {(1-n_features/len(el_feature_coef))*100:2.2f}%')
el_feature_coef.sort_values().plot(kind = 'bar', figsize = (13,5));
```
image 9

In terms of feature importance, the elastic model reduced features by 72%. Even with this reduction, the model does not seem to give an improved score against the ridge or lasso model. This is probably due to the small dataset and the linear models tendency to overfit.

KNN
For KNN, we will use the KNeighborsRegressor from sklearn. We apply the same process to find the optimal neighbor parameter.
```
from sklearn.neighbors import KNeighborsRegressor

params_knn = {
    'n_neighbors' : [3, 5, 7, 9, 11],
}

knn_model = KNeighborsRegressor()
knn_regressor = GridSearchCV(knn_model, params_knn, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
knn_regressor.fit(X_train, y_train)
print(f'Optimal neighbors: {knn_regressor.best_params_["n_neighbors"]}')
print(f'Best score: {knn_regressor.best_score_}')
```
result:
Optimal neighbors: 11
Best score: -0.43011660772832583

```
knn_model = KNeighborsRegressor(n_neighbors=knn_regressor.best_params_["n_neighbors"])
knn_model.fit(X_train, y_train)
y_train_pred = knn_model.predict(X_train)
y_pred = knn_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```
result:
Train r2 score:  -7.188173431615837
Test r2 score:  0.1434305705894996
Train RMSE: 0.4071
Test RMSE: 0.5115

Surprisingly, the KNN model seems to perform a bit better than the linear models on the test set.
Random Forest
Random forests are very powerful models which are a bit different from bagged decision trees. Unlike bagged trees, random forests will select a subset of features at random and finds the best feature to split at each node whereas bagged trees considers using all features for splitting at each node. Random forests also provide unique hyperparameters to reduce overfitting as well. We will tune our model based on several of these hyperparameters.

```
from sklearn.ensemble import RandomForestRegressor

params_rf = {
    'max_depth': [10, 30, 35, 50, 65, 75, 100],
    'max_features': [.3, .4, .5, .6],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [30, 50, 100, 200]
}

rf = RandomForestRegressor()
rf_regressor = GridSearchCV(rf, params_rf, scoring='neg_root_mean_squared_error', cv = 10, n_jobs = -1)
rf_regressor.fit(X_train, y_train)
print(f'Optimal depth: {rf_regressor.best_params_["max_depth"]}')
print(f'Optimal max_features: {rf_regressor.best_params_["max_features"]}')
print(f'Optimal min_sample_leaf: {rf_regressor.best_params_["min_samples_leaf"]}')
print(f'Optimal min_samples_split: {rf_regressor.best_params_["min_samples_split"]}')
print(f'Optimal n_estimators: {rf_regressor.best_params_["n_estimators"]}')
print(f'Best score: {rf_regressor.best_score_}')
```
result::
Optimal depth: 10
Optimal max_features: 0.3
Optimal min_sample_leaf: 4
Optimal min_samples_split: 8
Optimal n_estimators: 50
Best score: -0.4000337080466469

```
rf_model = RandomForestRegressor(max_depth=rf_regressor.best_params_["max_depth"], 
                                 max_features=rf_regressor.best_params_["max_features"], 
                                 min_samples_leaf=rf_regressor.best_params_["min_samples_leaf"], 
                                 min_samples_split=rf_regressor.best_params_["min_samples_split"], 
                                 n_estimators=rf_regressor.best_params_["n_estimators"], 
                                 n_jobs=-1, oob_score=True)
rf_model.fit(X_train, y_train)
y_train_pred = rf_model.predict(X_train)
y_pred = rf_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```
result:
Train r2 score:  -0.6757065673586051
Test r2 score:  0.19980759117131874
Train RMSE: 0.2795
Test RMSE: 0.4944

We can see the training score has improved significantly compared to the linear models we’ve used above and KNN. The model is also able to achieve this using nearly all of the data as shown below in the feature importance plot.
```
# Random Forest Model Feature Importance
rf_feature_importance = pd.Series(index = X_train.columns, data = np.abs(rf_model.feature_importances_))
n_features = (rf_feature_importance>0).sum()
print(f'{n_features} features with reduction of {(1-n_features/len(rf_feature_importance))*100:2.2f}%')
rf_feature_importance.sort_values().plot(kind = 'bar', figsize = (13,5));
```
image 11

LightGBM
LightGBM provides boosting capabilities for decision trees. It is a good alternative to XGBoost which we will test as well after.

```
import lightgbm as lgbm

params_lgbm = {
    'learning_rate': [.01, .1, .5, .7, .9, .95, .99, 1],
    'boosting': ['gbdt'],
    'metric': ['l1'],
    'feature_fraction': [.3, .4, .5, 1],
    'num_leaves': [20],
    'min_data': [10],
    'max_depth': [10],
    'n_estimators': [10, 30, 50, 100]
}

lgb = lgbm.LGBMRegressor()
lgb_regressor = GridSearchCV(lgb, params_lgbm, scoring='neg_root_mean_squared_error', cv = 10, n_jobs = -1)
lgb_regressor.fit(X_train, y_train)
print(f'Optimal lr: {lgb_regressor.best_params_["learning_rate"]}')
print(f'Optimal feature_fraction: {lgb_regressor.best_params_["feature_fraction"]}')
print(f'Optimal n_estimators: {lgb_regressor.best_params_["n_estimators"]}')
print(f'Best score: {lgb_regressor.best_score_}')
```
result:
Optimal lr: 0.1
Optimal feature_fraction: 0.4
Optimal n_estimators: 50
Best score: -0.38394173043287

```
lgb_model = lgbm.LGBMRegressor(learning_rate=lgb_regressor.best_params_["learning_rate"], boosting='gbdt', 
                               metric='l1', feature_fraction=lgb_regressor.best_params_["feature_fraction"], 
                               num_leaves=20, min_data=10, max_depth=10, 
                               n_estimators=lgb_regressor.best_params_["n_estimators"], n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_train_pred = lgb_model.predict(X_train)
y_pred = lgb_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```
result:
Train r2 score:  0.8402215166975928
Test r2 score:  0.1972437367117208
Train RMSE: 0.1425
Test RMSE: 0.4952

Based on the training score, it seems that the model has overfitted to the training set since there is no improvement in the test score. It’s probably not optimal to include this model at all in our ensemble later.

```
# LightGBM Feature Importance
lgb_feature_importance = pd.Series(index = X_train.columns, data = np.abs(lgb_model.feature_importances_))
n_features = (lgb_feature_importance>0).sum()
print(f'{n_features} features with reduction of {(1-n_features/len(lgb_feature_importance))*100:2.2f}%')
lgb_feature_importance.sort_values().plot(kind = 'bar', figsize = (13,5));
```
image 12

XGBoost
XGBoost is yet another boosting algorithm for decision trees. Let’s see how it compares to LightGBM after tuning hyperparameters.
```
params_xgb = {
    'learning_rate': [.1, .5, .7, .9, .95, .99, 1],
    'colsample_bytree': [.3, .4, .5, .6],
    'max_depth': [4],
    'alpha': [3],
    'subsample': [.5],
    'n_estimators': [30, 70, 100, 200]
}

xgb_model = XGBRegressor()
xgb_regressor = GridSearchCV(xgb_model, params_xgb, scoring='neg_root_mean_squared_error', cv = 10, n_jobs = -1)
xgb_regressor.fit(X_train, y_train)
print(f'Optimal lr: {xgb_regressor.best_params_["learning_rate"]}')
print(f'Optimal colsample_bytree: {xgb_regressor.best_params_["colsample_bytree"]}')
print(f'Optimal n_estimators: {xgb_regressor.best_params_["n_estimators"]}')
print(f'Best score: {xgb_regressor.best_score_}')
```
result:
Optimal lr: 0.95
Optimal colsample_bytree: 0.5
Optimal n_estimators: 100
Best score: -0.3992164151795667

```
xgb_model = XGBRegressor(learning_rate=xgb_regressor.best_params_["learning_rate"], 
                         colsample_bytree=xgb_regressor.best_params_["colsample_bytree"], 
                         max_depth=4, alpha=3, subsample=.5, 
                         n_estimators=xgb_regressor.best_params_["n_estimators"], n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_train_pred = xgb_model.predict(X_train)
y_pred = xgb_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```
result:
Train r2 score:  0.5795086544898738
Test r2 score:  0.32589181766269526
Train RMSE: 0.2309
Test RMSE: 0.4538

The model doesn’t seem to be overfitting as much as LightGBM. The training and test scores seem to be lower than the random forest model too. This explains why the model is heavily used in just about any problem setting.

```
# XGB with early stopping
xgb_model.fit(X_train, y_train, early_stopping_rounds=4,
             eval_set=[(X_test, y_test)], verbose=False)
y_train_pred = xgb_model.predict(X_train)
y_pred = xgb_model.predict(X_test)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
```
result:
Train r2 score:  -0.15853608490123916
Test r2 score:  0.3050460256750407
Train RMSE: 0.3131
Test RMSE: 0.4607

It is very easy to overfit in boosted models so we will add early stopping parameters to reduce overfitting. This gives us a better model that is still able to generalize the test set quite well.

```
# XGB Feature Importance, relevant features can be selected based on its score
feature_important = xgb_model.get_booster().get_fscore()
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=['score']).sort_values(by = 'score', ascending=True)
data.plot(kind='bar', figsize = (13,5))
plt.show()
```
 image 13
 
Ensembling
Based on the experimentation of the models above, it’s clear that the linear models and KNN are not the best models for this dataset. Therefore, they won’t be used as part of the ensemble. The best models to ensemble would be random forests and XGBoost models as we have seen from the training and test errors above. I decided to use a random forest ensemble since boosting models in this scenario have a tendency to overfit as shown by the LightGBM model. For the ensemble, I decided to use a stacked ensemble. The benefits of this is to create a single model that has the well-performing capabilities of several base models. The base models are different tuned random forest models and the meta model will be a simple model such as a linear regressor.
 
```
# define the base models
base_model = list()
base_model.append(('rf1', rf_model))
base_model.append(('rf2', rf_model_en))
base_model.append(('rf3', RandomForestRegressor(max_depth=8, max_features=0.1, min_samples_leaf=3, 
                                                min_samples_split=2, n_estimators=250, n_jobs=-1, oob_score=False)))
# define meta learner model
learner = LinearRegression()
# define the stacking ensemble
stack2 = StackingRegressor(estimators=base_model, final_estimator=learner, cv=10)
# fit the model on all available data
stack2.fit(X, y)
StackingRegressor(cv=10,
                  estimators=[('rf1',
                               RandomForestRegressor(max_depth=10,
                                                     max_features=0.3,
                                                     min_samples_leaf=4,
                                                     min_samples_split=8,
                                                     n_estimators=50, n_jobs=-1,
                                                     oob_score=True)),
                              ('rf2',
                               RandomForestRegressor(max_depth=200,
                                                     max_features=0.4,
                                                     min_samples_leaf=3,
                                                     min_samples_split=6,
                                                     n_estimators=30, n_jobs=-1,
                                                     oob_score=True)),
                              ('rf3',
                               RandomForestRegressor(max_depth=8,
                                                     max_features=0.1,
                                                     min_samples_leaf=3,
                                                     n_estimators=250,
                                                     n_jobs=-1))],
                  final_estimator=LinearRegression())
                 
```

I fitted the stacked model on the entire dataset and tested it against the Kaggle private leaderboard. The model did surprisingly well placing 4th on the private leaderboard with a RMSE score of 1741680.77896. For reference, the 1st place solution on the private leaderboard was an RMSE of 1727811.48553.

Scores
image 14

Challenges & Lessons Learned
Initially, I had not planned for a lot of things for this project. Before I read some of the discussions in the Kaggle competition, I had assumed the p-variables were actually numeric values. With this in mind, I log transformed these variables which actually ended up making the linear models better in predicting the target variable. This makes sense since every feature is normalized making it easier for a linear model to predict. After finding out that the p-variables were categorical with many missing values, imputation was a much better approach. This brings up the lesson that it’s very important to understand the data you’re working with especially if it’s a small dataset. I had also played around with many models, manually tweaking hyperparameters until I discovered grid search which did wonders on saving time and effort in finding the best model. These were just some of the few things I learned while working on this dataset. A few things to perhaps try in the future would be to fit models on relevant features and including a more diverse ensemble of












