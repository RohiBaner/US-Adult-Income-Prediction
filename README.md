# Readme File for Prediction of Income for 1994 Census Database (Python 3) #
The given training and testing dataset is from the 1994 Census Database. The task is to predict whether a person makes above 50K a year or not. 


## 1. Getting Started ##
This assignment has used ensemble learning method for the above stated classification problem. In particular, CatBoost Classifier from Yandex Technologies has been used.

### 1.1. Data Description ###
1. **trainFeatures.csv** - 34189 individual’s basic information with 14 attributes for training.
2. **testFeatures.csv** - 14653 individual’s basic information with 14 attributes for testing.
3. **trainLabels.csv** - 34189 individual’s incomes for training; 0: <=50k, 1: >50k.

### 1.2. Prerequisites ###
I have used **numpy**, **pandas**, **sklearn** and **catboost** libraries for this prediction problem. While the others are usually available when using Python3 via Anaconda, **catboost** needs to be downloaded separately. Please make sure the above mentioned libraries are present before running this code.

## 2. Methodology
### 2.1. Data Reading and Cleaning Stage ###
In these two minor stages, the three above mentioned data files are read into the code as *pandas* dataframes. Afterwards, the training attributes and label dataframes are merged together (as *data*) for easier preprocessing.

All duplicate rows are then removed from the *data* dataframe for initial cleaning.

### 2.2 Data Preprocessing Stage ###
Upon preliminary review, it is found that certain elements in the *workclass*, *occupation* and *native-country* columns have missing values (i.e., *?*). To take care of this, certain data preprocessing is required.

Initially, the *?* is taken to be a separate class altogether. However, the final accuracy is not very high with this method. So, I tried to predict the missing values of each class via Decision Tree and KNN. This method did provide better results, but had a much longer time complexity.

Ultimately, I decided to go with few other preprocessing techniques after analyzing the correlation between attributes. They are as follows:
1. If *age* is less than 18, *workclass* is made to be *Never-worked*.
2. If a person has never worked, it is obvious that his/her *occupation* should be none of the pre-defined classes. Hence it is put as *None*.
3. Any other missing value (*?*) in the *workclass* and *occupation* column is replaced with the modal value (most frequent value) of that column.
4. In the *native-country* column, it is seen that *United-States* heavily overshadows all other classes combined together. Hence, the missing values (*?*) are taken to be a part of the majority class, i.e. *United-States*. Also, all other country classes are combined to *Others*.

### 2.3. Feature Engineering Stage ###
Even though initial Gini Index of the features showed that only three attributes (*age*, *workclass* and *fnlwgt*) are important, the resultant model gave very low performance measures. In fact, elimination of any attribute led to diminished performance. Hence, not many features could be dropped.

In the end, I considered merging *capital-gain* and *capital-loss* into a feature called *total_cap*. This allowed me to drop the *capital-gain* and *capital-loss* attributes and normalize the continuous *total_cap* attribute. I also normalized the *fnlwgt* attribute. Since the *education-num* attribute was the ordinal label of *education* attribute, I dropped the *education* attribute as well.

As all the remaining categorical variables were nominal in nature, I performed one-hot encoding (and dropped the first one-hot column to prevent the attributes from becoming inter-dependent).

The above data preprocessing and feature engineering stages (except removing duplicates) were carried out on both training and test data.

### 2.4. Training Stage ###
For this assignment, I have used 10 fold cross validation technique with CatboostClassifier with a learning rate of 0.05. Precision, recall, f1-score and accuracy were chosen to be performance measures. This model provided an average accuracy of 87.3%. This was higher than what I got with XGBoost (86.2%), AdaBoost (85.9%) and RandomForest (84.7%). Hence, I chose to go with this model.

### 2.5. Testing Stage ###
The test features are provided to the trained model for predicting the output classes for the test set. The received predictions are then saved into a CSV file.

## 3. Authors ##
BANERJEE, Rohini - HKUST Student ID: 20543577
