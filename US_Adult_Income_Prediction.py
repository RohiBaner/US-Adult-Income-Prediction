"""@author: BANERJEE, Rohini (Student ID: 20543577)
MSBD5002: Data Mining and Knowledge Discovery (Assignment 2)
Title: Income Prediction for 1994 Census Database"""


# Import all required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from catboost import CatBoostClassifier
# Ignore any deprecation warning for a particular library version
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''1. DATASET READING STAGE'''
# Read the training features, training labels and testing features
X = pd.read_csv('trainFeatures.csv')
Y = pd.read_csv('trainLabels.csv', header = None)
Test = pd.read_csv('testFeatures.csv')
Y.columns = ["Label"] # add header to training label CSV file
# Concatenate training features and labels for easier preprocessing
data = pd.concat([X, Y], axis=1, sort=False)

'''2. DATA CLEANING STAGE'''
# Drop any duplicate rows from the training set
data.drop_duplicates(keep='first',inplace=True)

'''3. DATA PREPROCESSING STAGE'''
# If 'age' is less than 18, make 'workclass' attribute as 'Never worked'
data.loc[data['age'] < 18,'workclass'] = ' Never-worked'
Test.loc[Test['age'] < 18,'workclass'] = ' Never-worked'
# If a person has never worked, his occupation should be 'None'
data.loc[data['workclass']==' Never-worked','occupation'] = 'None'
Test.loc[Test['workclass']==' Never-worked','occupation'] = 'None'
# Replace missing values of 'workclass' with the most frequent value
data.loc[data['workclass'] == ' ?', 'workclass'] = data['workclass'].value_counts().idxmax()
Test.loc[Test['workclass'] == ' ?', 'workclass'] = Test['workclass'].value_counts().idxmax()
# Replace missing values of 'occupation' with the most frequent value
data.loc[data['occupation']==' ?','occupation'] = data['occupation'].value_counts().idxmax()
Test.loc[Test['occupation']==' ?','occupation'] = Test['occupation'].value_counts().idxmax()
# Since 'United States' heavily dominates the 'country' attribute,
# replace all missing values to 'United States'.
# Also, merge all other countries into a separate group
data.loc[data['native-country']==' ?','native-country'] = ' United-States'
data.loc[data['native-country']!=' United-States','native-country'] = ' Others'
Test.loc[Test['native-country']==' ?','native-country'] = ' United-States'
Test.loc[Test['native-country']!=' United-States','native-country'] = ' Others'

'''4. FEATURE ENGINEERING STAGE'''
# Merge the 'capital gain' and 'capital loss' into 'total capital'
data['total_cap'] = data['capital-gain'] - data['capital-loss']
Test['total_cap'] = Test['capital-gain'] - Test['capital-loss']
# Normalize the 'total capital' attribute
data['total_cap'] = (data['total_cap'] - data['total_cap'].mean())/data['total_cap'].std()
Test['total_cap'] = (Test['total_cap'] - Test['total_cap'].mean())/Test['total_cap'].std()
# Normalize the 'fnlwgt' attribute
data['fnlwgt'] = (data['fnlwgt'] - data['fnlwgt'].mean())/data['fnlwgt'].std()
Test['fnlwgt'] = (Test['fnlwgt'] - Test['fnlwgt'].mean())/Test['fnlwgt'].std()
# Drop the attributes which are no longer required
# 'education' dropped as 'education-num' is a labelled version of that column itself
data = data.drop(columns=['education','capital-gain', 'capital-loss'])
Test = Test.drop(columns=['education','capital-gain', 'capital-loss'])
# All nominal categorical variables are one-hot encoded
data = pd.get_dummies(data,drop_first=True)
Test = pd.get_dummies(Test,drop_first=True)

'''5. TRAINING STAGE'''
Y = data['Label'] # Taking only the training label
X = data.drop(columns = ['Label']) # Taking all input features only
X = X.values
Y = Y.values
Y = Y.reshape(len(Y))
# Few lists for calculating the mean model performance
accuracy = []
f1 = []
precision=[]
recall=[]
# 10-Fold cross validation on the training set
for train, test in KFold(n_splits=10).split(X):
    x_train, x_cv = X[train], X[test]
    y_train, y_cv = Y[train], Y[test]
    # Using CatBoost Classifier for the model
    model = CatBoostClassifier(learning_rate=0.05)
    model.fit(x_train, y_train) # Fit the model on the training set
    pred = model.predict(x_cv) # Validate the model on the cross-validation set
    # Calculate few model performance measures during each fold
    accuracy.append(accuracy_score(y_cv,pred))
    precision.append(precision_score(y_cv, pred))
    recall.append(recall_score(y_cv, pred))
    f1.append(f1_score(y_cv, pred))
# Print the average value of performance measures after 10-Fold cross-validation
print("Average Recall:",sum(recall)/len(recall))
print("Average Precision:",sum(precision)/len(precision))
print("Average Accuracy:",sum(accuracy)/len(accuracy))
print("Average F1:",sum(f1)/len(f1))

'''6. TESTING PHASE'''
Test = Test.values
pred = model.predict(Test) # Predict model for testing data
np.array(pred)
pred = pred.reshape(len(pred),1)
# Save predicted output in a CSV file
np.savetxt("prediction.csv", pred, delimiter=",", fmt='%10.0f')
