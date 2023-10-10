#!/usr/bin/env python
# coding: utf-8

# # 5. Deployment

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Parameters
C = 1.0
output_file = f'model_C={C}.bin'
n_splits = 5



# Data preparation
df = pd.read_csv('data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# Data Split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# Training
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

dv, model = train(df_train, y_train, C=0.001)

# Predicting
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

y_pred = predict(df_val, dv, model)


# Validation

print(f'doing validation with C={C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0
for train_idx, val_idx in kfold.split(df_full_train):
    #get training and validation data based on k-fold indexes
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    #get target(y) values for train and val dataset
    y_train = df_train.churn.values
    y_val = df_val.churn.values

    #train modelwith splitted dataset and then predict
    #use parameter based on "C" variable (here we check various "C" parameter values)
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    #calculate auc and append it to scores list
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))




# Train Final Model
print('training the final model')
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

# calculate auc score
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# #### Save the model

#we do write binary file process and will contain 2 things: dv & model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')

#automatically close file when outside "with"





