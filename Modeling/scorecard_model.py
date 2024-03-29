# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:10:34 2020

@author: Nicholas Sotiriou - github: @nsotiriou88 // nsotiriou88@gmail.com
"""

# Traditional Credit Scoring Using Logistic Regression
import scorecardpy as sc

# data prepare ------
# load germancredit data
dat = sc.germancredit()

# filter variable via missing rate, iv, identical value rate
dt_s = sc.var_filter(dat, y="creditability")

# breaking dt into train and test
train, test = sc.split_df(dt_s, 'creditability').values()


#%% woe binning ------
bins = sc.woebin(dt_s, y="creditability")
# sc.woebin_plot(bins)


#%% binning adjustment
# # adjust breaks interactively
# breaks_adj = sc.woebin_adj(dt_s, "creditability", bins) 
# # or specify breaks manually
breaks_adj = {
    'age.in.years': [26, 35, 40],
    'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
}
bins_adj = sc.woebin(dt_s, y="creditability", breaks_list=breaks_adj)


#%% converting train and test into woe values
train_woe = sc.woebin_ply(train, bins_adj)
test_woe = sc.woebin_ply(test, bins_adj)

y_train = train_woe.loc[:,'creditability']
X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe.loc[:,'creditability']
X_test = test_woe.loc[:,train_woe.columns != 'creditability']


#%% logistic regression ------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
# lr.coef_
# lr.intercept_

# predicted proability
train_pred = lr.predict_proba(X_train)[:,1]
test_pred = lr.predict_proba(X_test)[:,1]

# performance ks & roc ------
train_perf = sc.perf_eva(y_train, train_pred, title = "train")
test_perf = sc.perf_eva(y_test, test_pred, title = "test")


#%% score ------
card = sc.scorecard(bins_adj, lr, X_train.columns)
# credit score
train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)


#%% psi
sc.perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}
)


#%% xgboost ------
from xgboost import XGBClassifier
classifier = XGBClassifier() #can change learning rate etc.
classifier.fit(X_train, y_train)

# predicted proability
train_pred2 = classifier.predict_proba(X_train)[:,1]
test_pred2 = classifier.predict_proba(X_test)[:,1]

# performance ks & roc ------
train_perf2 = sc.perf_eva(y_train, train_pred2, title = "train")
test_perf2 = sc.perf_eva(y_test, test_pred2, title = "test")
