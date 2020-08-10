# Grid Search

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset using CSV
dataset = pd.read_csv('YOUR_CSV_DATASET.csv')
X = dataset.iloc[:, [2, 3]].values #independent variable(s)
y = dataset.iloc[:, 4].values #dependent variable



# Splitting the dataset into the Training set and Test set
# Change proportion or maybe you want to test it on a different dataset?????
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#Random state helps having the same results when splitting the sets everytime.



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#############################################
##### SVC with Gausian Kernel Example #######
#############################################
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, #we did not investigate the polynomial kernel; need degree parameter for it
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy', #validation metric or remove for the default scorer for the classifier
                           cv = 10, #numbers of cross validation per model, per set of parameters
                           n_jobs = -1) #using all cores!!! or specify the number of cores you want to use
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy:", best_accuracy)
print("Best Parameters:", best_parameters)
#It also includes other methods and you can print all results too

#############################################
#############################################


#############################################
###### Logistic Regression Classifier #######
#############################################
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifierLog = LogisticRegression()
classifierLog.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifierLog.predict(X_test)
# Making the Confusion Matrix (checking the correlation between prediction and actual test data)
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
# For the full of parameters to tweak, please inspect the module LogisticRegression(Ctrl+I on Spyder)
parameters = [{'C': [1, 10, 100, 1000], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
grid_search = GridSearchCV(estimator = classifierLog,
                           param_grid = parameters,
                           scoring = 'accuracy', #validation metric or remove for the default scorer for the classifier
                           cv = 10, #numbers of cross validation per model, per set of parameters
                           n_jobs = -1) #using all cores!!!
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy:", best_accuracy)
print("Best Parameters:", best_parameters)

#############################################
#############################################

