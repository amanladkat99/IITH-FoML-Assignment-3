import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

#Data paths are given as per file locations in PC
original_data = pd.read_csv('D:\MTECH\Sem-1\Foundations of ML\loan_train.csv')

#Selected columns
selected_columns = ['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','emp_length','home_ownership','annual_inc','verification_status','dti','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','revol_bal','total_acc','total_rec_int','total_rec_late_fee','recoveries','loan_status']

training_data = original_data.loc[:, selected_columns]
training_data = training_data.loc[training_data['loan_status'].isin(['Fully Paid','Charged Off'])]

#Missing values replace with mode
training_data['emp_length'] = training_data['emp_length'].fillna(training_data['emp_length'].mode()[0])

labels = {'loan_status':{'Charged Off':-1,'Fully Paid':1}}
training_data = training_data.replace(labels)

training_data['int_rate'] = training_data['int_rate'].str.strip('%')
training_data['int_rate'] = training_data.int_rate.astype(float)

labels = {"emp_length": {'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10 years':10,'< 1 year':0, '10+ years':11 }}
training_data = training_data.replace(labels)

labels = {"home_ownership":{'RENT':1,'MORTGAGE':2,'OWN':3,'OTHER':4}}
training_data = training_data.replace(labels)

labels = {"verification_status":{'Not Verified':1,'Source Verified':2,'Verified':3}}
training_data = training_data.replace(labels)

labels = {"term":{' 36 months':36,' 60 months':60}}
training_data = training_data.replace(labels)

# Applying all pre-processing steps on test data.
original_data1 = pd.read_csv('D:\MTECH\Sem-1\Foundations of ML\loan_test.csv')
actual_columns1 = ['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','emp_length','home_ownership','annual_inc','verification_status','dti','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','revol_bal','total_acc','total_rec_int','total_rec_late_fee','recoveries','loan_status']
testing_data = original_data1.loc[:, actual_columns1]

testing_data = testing_data.loc[testing_data['loan_status'].isin(['Fully Paid','Charged Off'])]
testing_data['emp_length'] = testing_data['emp_length'].fillna(testing_data['emp_length'].mode()[0])

labels = {'loan_status':{'Charged Off':-1,'Fully Paid':1}}
testing_data = testing_data.replace(labels)

testing_data['int_rate'] = testing_data['int_rate'].str.strip('%')
testing_data['int_rate'] = testing_data.int_rate.astype(float)

labels = {"emp_length": {'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10 years':10,'< 1 year':0, '10+ years':11 }}
testing_data = testing_data.replace(labels)

labels = {"home_ownership":{'RENT':1,'MORTGAGE':2,'OWN':3,'OTHER':4,'NONE':5}}
testing_data = testing_data.replace(labels)

labels = {"verification_status":{'Not Verified':1,'Source Verified':2,'Verified':3}}
testing_data = testing_data.replace(labels)

labels = {"term":{' 36 months':36,' 60 months':60}}
testing_data = testing_data.replace(labels)

#Normalize the dataset
scaler = MinMaxScaler()
training_data=pd.DataFrame(scaler.fit_transform(training_data),columns=training_data.columns, index=training_data.index)
training_data
testing_data=pd.DataFrame(scaler.fit_transform(testing_data),columns=testing_data.columns, index=testing_data.index)
testing_data

df = pd.DataFrame(training_data)
X_train = df.iloc[: , :20]
df = pd.DataFrame(training_data)
Y_train = df.iloc[: , -1]
df = pd.DataFrame(testing_data)
X_test = df.iloc[: , :20]
df = pd.DataFrame(testing_data)
Y_test = df.iloc[: , -1]

print("**************************************************************************")
print("Model 1 with default values")
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, max_depth=3, max_features=None, random_state=None)
clf.fit(X_train,Y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(Y_test,predict)
print("Test Accuracy of Gradient Boosting algorithm is : " + str(round((accuracy),5)))
TP, FP, FN, TP = confusion_matrix(Y_test, predict).ravel()
print("Value of Precision is : " + str(round((TP)/(TP+FP),5)))
print("Value of Recall is : "+ str(round((TP)/(TP+FN),5)))

print("**************************************************************************")
print("Model 2")
clf = GradientBoostingClassifier(loss='exponential', learning_rate=1, n_estimators=10, max_depth=3, max_features=2, random_state=None)
clf.fit(X_train,Y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(Y_test,predict)
print("Test Accuracy of Gradient Boosting algorithm is : " + str(round((accuracy),5)))
TP, FP, FN, TP = confusion_matrix(Y_test, predict).ravel()
print("Value of Precision is : " + str(round((TP)/(TP+FP),5)))
print("Value of Recall is : "+ str(round((TP)/(TP+FN),5)))

print("**************************************************************************")
print("Model 3")
clf = GradientBoostingClassifier(loss='deviance', learning_rate=10, n_estimators=200, max_depth=5, max_features=2, random_state=None)
clf.fit(X_train,Y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(Y_test,predict)
print("Test Accuracy of Gradient Boosting algorithm is : " + str(round((accuracy),5)))
TP, FP, FN, TP = confusion_matrix(Y_test, predict).ravel()
print("Value of Precision is : " + str(round((TP)/(TP+FP),5)))
print("Value of Recall is : "+ str(round((TP)/(TP+FN),5)))

print("**************************************************************************")
print("Model 4")
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=500, max_depth=1, max_features=None, random_state=None)
clf.fit(X_train,Y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(Y_test,predict)
print("Test Accuracy of Gradient Boosting algorithm is : " + str(round((accuracy),5)))
TP, FP, FN, TP = confusion_matrix(Y_test, predict).ravel()
print("Value of Precision is : " + str(round((TP)/(TP+FP),5)))
print("Value of Recall is : "+ str(round((TP)/(TP+FN),5)))

print("**************************************************************************")
print("Decision Tree Implementation")
clf = DecisionTreeClassifier(criterion='gini',splitter='best',random_state=0)
clf.fit(X_train,Y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(Y_test,predict)
print("Test Accuracy of Decision Tree is : " + str(round((accuracy),5)))
TP, FP, FN, TP = confusion_matrix(Y_test, predict).ravel()
print("Value of Decision Tree Precision is : " + str(round((TP)/(TP+FP),5)))
print("Value of Decision Tree Recall is : "+ str(round((TP)/(TP+FN),5)))
print("**************************************************************************")



