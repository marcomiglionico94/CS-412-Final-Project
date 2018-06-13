#%% Secondo
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.metrics as mt
import xgboost as xgb
# from sklearn.ensemble import GradientBoostingClassifier as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import TransformerMixin

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

#%% Read data section
fileTrain = pd.read_csv("loan_clean.csv", sep =",")

print(fileTrain["loan_status"].value_counts())

# fileTrain = fileTrain.drop_duplicates(subset='ID')

new = [u'loan_amnt', u'term', u'int_rate', u'installment', u'emp_length',
       u'home_ownership', u'annual_inc', u'verification_status', u'purpose', u'zip_code', u'addr_state', u'dti',
       u'delinq_2yrs', u'inq_last_6mths', u'mths_since_last_delinq',
       u'mths_since_last_record', u'open_acc', u'pub_rec', u'revol_bal',
       u'revol_util', u'total_acc', u'initial_list_status',
       u'mths_since_last_major_derog', u'last_first_credit_diff']
new2 = [u'term', u'int_rate', u'emp_length', u'purpose', u'initial_list_status']

label = fileTrain.pop(u'loan_status')
df_clean_norm = (fileTrain-fileTrain.mean())/(fileTrain.max()-fileTrain.min())

X_train, X_test, y_train, y_test = ms.train_test_split(df_clean_norm[new],label, test_size = 0.3)
print(fileTrain.columns)
#%%

num_round = 50
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

evallist  = [(dtest,'eval'), (dtrain,'train')]
param = {'objective':'multi:softmax','base_score':0.5, 'silent':1, 'eval_metric': ['ndcg'],'eta':0.3,'gamma':30,'max_depth':7,
         'min_child_weight':20,'lambda':1,'alpha':0.5,'scale_pos_weight':1,'updater':'grow_local_histmaker,prune' }


param1 = {'objective':'binary:logistic','base_score':0.5, 'silent':1, 'eval_metric': ['ndcg'],'eta':0.5,'gamma':25,'max_depth':7,
         'min_child_weight':1,'lambda':0.5,'alpha':0.2,'scale_pos_weight':1,'updater':'grow_local_histmaker,prune' }

bst = xgb.train( param1, dtrain, num_round, evallist )
# bst = MultinomialNB(alpha=0.1)
# bst.fit(X_train, y_train)

y_predProb1 = bst.predict(dtest)
#%%
threshold = 0.20
y_pred1 = y_predProb1[:].copy()
y_pred1[y_pred1> threshold] = 1
y_pred1[y_pred1<= threshold] = 0

# y_pred1 = bst.predict(X_test)


print(mt.f1_score(y_test,y_pred1))
print(mt.accuracy_score(y_test,y_pred1))
print(mt.confusion_matrix(y_test,y_pred1))
