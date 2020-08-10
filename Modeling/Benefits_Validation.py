# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:27:24 2019

@author: Nicholas Sotiriou - github: @nsotiriou88
"""

# Testing the bad rate
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import recall_score, roc_auc_score


#%% Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


#%% Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)


#%% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#%% Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
y_prob_0 = y_prob[:, 0]
y_prob_1 = y_prob[:, 1]


#%% Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Plot confusion matrix
fig, ax = plt.subplots()
ax.set_aspect('equal')
hist, xbins, ybins, im = ax.hist2d([0,0,1,1], [0,1,0,1], bins=2,
weights=[cm[0,0],cm[0,1],cm[1,0],cm[1,1]], cmin=0, cmax=60,
cmap='PuBu')
plt.title('Confusion Matrix')
for i in range(len(ybins)-1):
    for j in range(len(xbins)-1):
        ax.text(xbins[j]+0.25,ybins[i]+0.25,int(hist[i,j]),color='black',
        ha='center',va='center', fontweight='bold')
ax.set_xticks([0.25, 0.75])
ax.set_yticks([0.25, 0.75])
ax.set_xticklabels(['Negative','Positive'])
ax.set_yticklabels(['Negative','Positive'])
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
fig.colorbar(im)
plt.show()


#%% AUC plot
plt.figure()
y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
AUC = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, 'r-', label='ROC curve, area='+str(round(AUC,3)))
plt.plot([0,1],[0,1], 'c--')
plt.legend(loc=4)
plt.show()

total = np.sum(cm)

# Accuracy
accuracy = (tp+tn)/total*100
print('accuracy:', accuracy, '%')

# Recall/Sensitivity/TPR
recall = tp/(tp+fn)*100
print('Recall:', accuracy, '%')

# Precision
precision = tp/(tp+fp)*100
print('precision:', precision, '%')

# Specificity
specificity = tn/(tn+fp)*100
print('specificity:', specificity, '%')

# F-Score
f_score = 2*recall*precision/(recall+precision)
print('F-score:', f_score, '%')

gini = (2*AUC-1)*100
print('gini', gini, '%')


#%% Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#%% Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.savefig('python-naiveBayes.png')
plt.show()

