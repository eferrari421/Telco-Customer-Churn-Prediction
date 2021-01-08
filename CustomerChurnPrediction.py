#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 23:52:38 2020

@author: erikferrari
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import minmax_scaling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score


telco = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# DATA CLEANING

telco.columns

telco.head()

telco.dtypes

telco.info()

telco.shape

telco.nunique()

summary = telco.describe()

#removing any duplicate values
telco.drop_duplicates(inplace=True)
#no duplicates

#check for null values
telco.isnull().any()
#no null values

#but, there are values in the TotalCharges column that are empty
telco[telco.TotalCharges == ' ']

#we will replace with NaN
telco['TotalCharges'] = telco["TotalCharges"].replace(" ",np.nan)

telco.TotalCharges.isnull().sum()/telco.shape[0] * 100

#only .15% of TotalCharges data is null, so we will just drop it
telco = telco.dropna(axis=0)

#some values in TotalCharges are integers, convert all to float
telco['TotalCharges'] = telco.TotalCharges.astype(float)
telco['tenure'] = telco.tenure.astype(float)

#replace 'no internet service' and 'no phone service' to 'no' for the following columns
replace_columns = ['MultipleLines','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_columns:
    telco[i] = telco[i].replace({'No internet service':'No', 'No phone service':'No'})

#drop customerID column as it won't help with prediction
telco.drop('customerID', axis=1, inplace=True)


# DATA EXPLORATION AND VISUALIZATION


#first, let's look at the distribution of customer churn
sns.catplot(x='Churn', data=telco, kind='count', palette="Greens")
plt.title('Distribution of Customer Churn')

#let's look at the distribution of some other variables by churn
telco.groupby(['Churn', 'gender']).gender.count()
#and visualize it
sns.countplot(x='Churn', hue='gender', data=telco, palette='Greens')
plt.title('Distribution of Churn by Gender')

telco.groupby(['Churn', 'SeniorCitizen']).SeniorCitizen.count()
sns.countplot(x='Churn', hue='SeniorCitizen', data=telco, palette='Greens')
plt.title('Distribution of Churn by SeniorCitizen')

telco.groupby(['Churn', 'Partner']).Partner.count()
sns.countplot(x='Churn', hue='Partner', data=telco, palette='Greens')
plt.title('Distribution of Churn by Partner')

telco.groupby(['Churn', 'Dependents']).Dependents.count()
sns.countplot(x='Churn', hue='Dependents', data=telco, palette='Greens')
plt.title('Distribution of Churn by Dependents')

telco.groupby(['Churn', 'PhoneService']).PhoneService.count()
sns.countplot(x='Churn', hue='PhoneService', data=telco, palette='Greens')
plt.title('Distribution of Churn by PhoneService')

telco.groupby(['Churn', 'MultipleLines']).MultipleLines.count()
sns.countplot(x='Churn', hue='MultipleLines', data=telco, palette='Greens')
plt.title('Distribution of Churn by MultipleLines')

telco.groupby(['Churn', 'InternetService']).PhoneService.count()
sns.countplot(x='Churn', hue='InternetService', data=telco, palette='Greens')
plt.title('Distribution of Churn by InternetService')

telco.groupby(['Churn', 'OnlineSecurity']).OnlineSecurity.count()
sns.countplot(x='Churn', hue='OnlineSecurity', data=telco, palette='Greens')
plt.title('Distribution of Churn by OnlineSecurity')

telco.groupby(['Churn', 'OnlineBackup']).OnlineBackup.count()
sns.countplot(x='Churn', hue='OnlineBackup', data=telco, palette='Greens')
plt.title('Distribution of Churn by OnlineBackup')

telco.groupby(['Churn', 'DeviceProtection']).DeviceProtection.count()
sns.countplot(x='Churn', hue='DeviceProtection', data=telco, palette='Greens')
plt.title('Distribution of Churn by DeviceProtection')

telco.groupby(['Churn', 'TechSupport']).TechSupport.count()
sns.countplot(x='Churn', hue='TechSupport', data=telco, palette='Greens')
plt.title('Distribution of Churn by TechSupport')

telco.groupby(['Churn', 'StreamingTV']).StreamingTV.count()
sns.countplot(x='Churn', hue='StreamingTV', data=telco, palette='Greens')
plt.title('Distribution of Churn by StreamingTV')

telco.groupby(['Churn', 'StreamingMovies']).StreamingMovies.count()
sns.countplot(x='Churn', hue='StreamingMovies', data=telco, palette='Greens')
plt.title('Distribution of Churn by StreamingMovies')

telco.groupby(['Churn', 'Contract']).Contract.count()
sns.countplot(x='Churn', hue='Contract', data=telco, palette='Greens')
plt.title('Distribution of Churn by Contract')

telco.groupby(['Churn', 'PaperlessBilling']).PaperlessBilling.count()
sns.countplot(x='Churn', hue='PaperlessBilling', data=telco, palette='Greens')
plt.title('Distribution of Churn by PaperlessBilling')

telco.groupby(['Churn', 'PaymentMethod']).PaymentMethod.count()
sns.countplot(x='Churn', hue='PaymentMethod', data=telco, palette='Greens')
plt.title('Distribution of Churn by PaymentMethod')

tenure_yes = pd.DataFrame(telco.tenure.loc[telco['Churn']=='Yes'])
tenure_no = pd.DataFrame(telco.tenure.loc[telco['Churn']=='No'])
plt.title('Tenure by Churn')
plt.xlabel('Tenure')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=tenure_yes['tenure'], label='Churn-yes')
sns.kdeplot(data=tenure_no['tenure'], label='Churn-no')

monthly_charges_yes = pd.DataFrame(telco.MonthlyCharges.loc[telco['Churn']=='Yes'])
monthly_charges_no = pd.DataFrame(telco.MonthlyCharges.loc[telco['Churn']=='No'])
plt.title('Monthly Charges by Churn')
plt.xlabel('Monthly Charges')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=monthly_charges_yes['MonthlyCharges'], label='Churn-yes')
sns.kdeplot(data=monthly_charges_no['MonthlyCharges'], label='Churn-no')

total_charges_yes = pd.DataFrame(telco.TotalCharges.loc[telco['Churn']=='Yes'])
total_charges_no = pd.DataFrame(telco.TotalCharges.loc[telco['Churn']=='No'])
plt.title('Total Charges by Churn')
plt.xlabel('Total Charges')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=total_charges_yes['TotalCharges'], label='Churn-yes')
sns.kdeplot(data=total_charges_no['TotalCharges'], label='Churn-no')

#plotting relationships between numerical features, by churn
plt.title('Relationship Between Tenure and Monthly Charges')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
sns.scatterplot(x='tenure', y='MonthlyCharges', data=telco, hue='Churn')

plt.title('Relationship Between Tenure and Total Charges')
plt.xlabel('Tenure')
plt.ylabel('Total Charges')
sns.scatterplot(x='tenure', y='TotalCharges', data=telco, hue='Churn')

plt.title('Relationship Between Total Charges and Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', data=telco, hue='Churn')

#relationships between all numerical variables - checking for colinnearity
plt.title('Correlation Between All Variables')
sns.heatmap(data=telco.corr(), square=True , annot=True, cbar=True, cmap='Greens')


# FEATURE ENGINEERING


#add total charges to tenure ratio
telco['TC_to_ten_ratio'] = telco['TotalCharges']/telco['tenure']

#add dummy variables for tenure age groups
telco['lessthan12months'] = np.where(telco['tenure'] <= 12,1,0)
telco['between12and24months'] = np.where((telco['tenure'] > 12) & (telco['tenure'] <= 24),1,0)
telco['between24and48months'] = np.where((telco['tenure'] > 24) & (telco['tenure'] <= 48),1,0)
telco['between48and60months'] = np.where((telco['tenure'] > 48) & (telco['tenure'] <= 60),1,0)
telco['over60months'] = np.where(telco['tenure'] > 60,1,0)

#add total services, the sum of all services with the company
telco['totalservices'] = (telco[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']]=='Yes').sum(axis=1)


# DATA PREPROCESSING


#label encoding columns with only two values(excluding the ones we created)
(telco.dtypes == 'object') & (telco.nunique() == 2) 
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
l_e = LabelEncoder()
for i in binary_cols:
    telco[i] = l_e.fit_transform(telco[i])

#dummy variables for the rest of the categorical columns
(telco.dtypes == 'object') & (telco.nunique() > 2) 
multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
telco = pd.get_dummies(data=telco, columns=multi_cols)

#convert churn to 1/0
telco['Churn'] = telco.Churn.map(dict(Yes=1, No=0))

#test/train before scaling to prevent data leakage
x = telco.drop(['Churn'], axis=1)
y = telco.Churn
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=2, test_size=0.2)

#scale the numerical columns
scale_col = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TC_to_ten_ratio']
cols_to_be_scaled_train = pd.DataFrame(train_x[scale_col])
cols_to_be_scaled_test = pd.DataFrame(test_x[scale_col])
cols_to_be_scaled_train = minmax_scaling(cols_to_be_scaled_train, columns = scale_col)
cols_to_be_scaled_test = minmax_scaling(cols_to_be_scaled_test, columns = scale_col)  
#drop unscaled columns
train_x = train_x.drop(columns=scale_col, axis=1)
test_x = test_x.drop(columns=scale_col, axis=1)
#add in scaled columns
train_x = train_x.merge(cols_to_be_scaled_train,left_index=True,right_index=True,how = "left")
test_x = test_x.merge(cols_to_be_scaled_test,left_index=True,right_index=True,how = "left")


# MODEL TRAINING


#lists for model comparison
models = ['logistic regression', 'naive bayes', 'knn', 'decision tree', 'svm', 'random forest']
accuracy = []
precision = []
recall = []
f1 = []
roc_auc = []

#basic models

#logistic regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(train_x, train_y)
#5 fold cross-validated scores
cv_1 = cross_val_score(estimator=log_reg, X=train_x, y=train_y, cv=5)
print(cv_1)
#predicting churn and calculating various metrics
log_reg_pred = log_reg.predict(test_x)
print(classification_report(test_y, log_reg_pred))
log_reg_acc = accuracy_score(test_y, log_reg_pred)
log_reg_prec = precision_score(test_y, log_reg_pred, pos_label=1)
log_reg_recall = recall_score(test_y, log_reg_pred, pos_label=1)
log_reg_f1 = f1_score(test_y, log_reg_pred, pos_label=1)
log_reg_roc = roc_auc_score(test_y, log_reg_pred)
print('Accuracy Score:', log_reg_acc)
print('Area Under ROC Curve:', log_reg_roc)
#plot confusion matrix 
log_conf = plot_confusion_matrix(log_reg, test_x, test_y, cmap='Greens')
#append metrics to lists
accuracy.append(log_reg_acc)
precision.append(log_reg_prec)
recall.append(log_reg_recall)
f1.append(log_reg_f1)
roc_auc.append(log_reg_roc)

#naive bayes
bnb = BernoulliNB()
bnb.fit(train_x, train_y)
cv_2 = cross_val_score(estimator=bnb, X=train_x, y=train_y, cv=5)
print(cv_2)
bnb_pred = bnb.predict(test_x)
print(classification_report(test_y, bnb_pred))
bnb_acc = accuracy_score(test_y, bnb_pred)
bnb_prec = precision_score(test_y, bnb_pred, pos_label=1)
bnb_recall = recall_score(test_y, bnb_pred, pos_label=1)
bnb_f1 = f1_score(test_y, bnb_pred, pos_label=1)
bnb_roc = roc_auc_score(test_y, bnb_pred)
print('Accuracy Score:', bnb_acc)
print('Area Under ROC Curve:', bnb_roc)
gnb_conf = plot_confusion_matrix(bnb, test_x, test_y, cmap='Greens')
accuracy.append(bnb_acc)
precision.append(bnb_prec)
recall.append(bnb_recall)
f1.append(bnb_f1)
roc_auc.append(bnb_roc)

#knn 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, train_y)
cv_3 = cross_val_score(estimator=knn, X=train_x, y=train_y, cv=5)
print(cv_3)
knn_pred = knn.predict(test_x)
print(classification_report(test_y, knn_pred))
knn_acc = accuracy_score(test_y, knn_pred)
knn_prec = precision_score(test_y, knn_pred, pos_label=1)
knn_recall = recall_score(test_y, knn_pred, pos_label=1)
knn_f1 = f1_score(test_y, knn_pred, pos_label=1)
knn_roc = roc_auc_score(test_y, knn_pred)
print('Accuracy Score:', knn_acc)
print('Area Under ROC Curve:', knn_roc)
knn_conf = plot_confusion_matrix(knn, test_x, test_y, cmap='Greens')
accuracy.append(knn_acc)
precision.append(knn_prec)
recall.append(knn_recall)
f1.append(knn_f1)
roc_auc.append(knn_roc)

#decision tree
dec_tree = DecisionTreeClassifier(random_state=1)
dec_tree.fit(train_x, train_y)
cv_4 = cross_val_score(estimator=dec_tree, X=train_x, y=train_y, cv=5)
print(cv_4)
dec_tree_pred = dec_tree.predict(test_x)
print(classification_report(test_y, dec_tree_pred))
dec_tree_acc = accuracy_score(test_y, dec_tree_pred)
dec_tree_prec = precision_score(test_y, dec_tree_pred, pos_label=1)
dec_tree_recall = recall_score(test_y, dec_tree_pred, pos_label=1)
dec_tree_f1 = f1_score(test_y, dec_tree_pred, pos_label=1)
dec_tree_roc = roc_auc_score(test_y, dec_tree_pred)
print('Accuracy Score:', dec_tree_acc)
print('Area Under ROC Curve:', dec_tree_roc)
dec_tree_conf = plot_confusion_matrix(dec_tree, test_x, test_y, cmap='Greens')
accuracy.append(dec_tree_acc)
precision.append(dec_tree_prec)
recall.append(dec_tree_recall)
f1.append(dec_tree_f1)
roc_auc.append(dec_tree_roc)

#support vector machine
svm = SVC()
svm.fit(train_x, train_y)
cv_5 = cross_val_score(estimator=svm, X=train_x, y=train_y, cv=5)
print(cv_5)
svm_pred = svm.predict(test_x)
print(classification_report(test_y, svm_pred))
svm_acc = accuracy_score(test_y, svm_pred)
svm_prec = precision_score(test_y, svm_pred, pos_label=1)
svm_recall = recall_score(test_y, svm_pred, pos_label=1)
svm_f1 = f1_score(test_y, svm_pred, pos_label=1)
svm_roc = roc_auc_score(test_y, svm_pred)
print('Accuracy Score:', svm_acc)
print('Area Under ROC Curve:', svm_roc)
svm_conf = plot_confusion_matrix(svm, test_x, test_y, cmap='Greens')
accuracy.append(svm_acc)
precision.append(svm_prec)
recall.append(svm_recall)
f1.append(svm_f1)
roc_auc.append(svm_roc)

#random forest
rand_for = RandomForestClassifier(random_state=1)
rand_for.fit(train_x, train_y)
cv_6 = cross_val_score(estimator=rand_for, X=train_x, y=train_y, cv=5)
print(cv_6)
rand_for_pred = rand_for.predict(test_x)
print(classification_report(test_y, rand_for_pred))
rand_for_acc = accuracy_score(test_y, rand_for_pred)
rand_for_prec = precision_score(test_y, rand_for_pred, pos_label=1)
rand_for_recall = recall_score(test_y, rand_for_pred, pos_label=1)
rand_for_f1 = f1_score(test_y, rand_for_pred, pos_label=1)
rand_for_roc = roc_auc_score(test_y, rand_for_pred)
print('Accuracy Score:', rand_for_acc)
print('Area Under ROC Curve:', rand_for_roc)
rand_for_conf = plot_confusion_matrix(rand_for, test_x, test_y, cmap='Greens')
accuracy.append(rand_for_acc)
precision.append(rand_for_prec)
recall.append(rand_for_recall)
f1.append(rand_for_f1)
roc_auc.append(rand_for_roc)

#let's use this random forest model to look at feature importances
feat_importances = pd.Series(rand_for.feature_importances_, index=train_x.columns)
feat_importances.nlargest(20).plot(kind='barh')

#now let's compare models
compare = pd.DataFrame({'Algorithms' : models , 'Accuracy' : accuracy, 'Precision': precision, 'Recall' : recall, 'F1': f1, 'ROC-AUC': roc_auc})
#different models perform better looking at different metrics
#class imbalance, so shouldn't look at accuracy, let's base on f1
compare.sort_values(by='F1', ascending=False)

#optimize hyper-parameters for naive bayes
param_grid={'alpha':[0.01, 0.1, 0.5, 1.0, 10.0]}
cv_nb = GridSearchCV(estimator=BernoulliNB(), param_grid=param_grid, cv=5)
cv_nb.fit(train_x, train_y)
cv_nb.best_params_
cv_nb_pred = cv_nb.predict(test_x)
print(classification_report(test_y, cv_nb_pred))
cv_nb_acc = accuracy_score(test_y, cv_nb_pred)
cv_nb_prec = precision_score(test_y, cv_nb_pred, pos_label=1)
cv_nb_recall = recall_score(test_y, cv_nb_pred, pos_label=1)
cv_nb_f1 = f1_score(test_y, cv_nb_pred, pos_label=1)
cv_nb_roc = roc_auc_score(test_y, cv_nb_pred)
print('Accuracy Score:', cv_nb_acc)
print('Area Under ROC Curve:', cv_nb_roc)
print('Precision:', cv_nb_prec)
print('Recall:', cv_nb_recall)
print('F1 Score:', cv_nb_f1)
cv_nb_conf = plot_confusion_matrix(cv_nb, test_x, test_y, cmap='Greens')


















