# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #Trick for ignoring double quotes (#3)

# Cleaning the texts
# Remove unecessary words like 'the', 'is' etc and NO capitals.
# Apply stemming --> Find roots of words so that we do not have to many to process
# e.g. 'loved' will be replaced by 'love'.
# Important NLT packages
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


## This is for removing the HTML tags for online text scrapping
#def cleanhtml(raw_html):
#    '''cleanhtml is removing the HTML tags from the text.'''
#    cleanr = re.compile('<.*?>')
#    cleantext = re.sub(cleanr, '', raw_html)
#    return cleantext

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # ^ means that we do not remove!!!
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #sets boost performance compared to slow lists
    review = ' '.join(review) #Space added for joining them
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Cleaning options can be set here for the text!!!
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set -- Most common alongside with Decision Trees
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#========

# Fitting Decision Tree Classification to the Training set -- Second Most Common
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#========

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Metrics about results
accuracy = (cm[0][0]+cm[1][1])*100/(np.sum(cm)) # (TP + TN) / (TP + TN + FP + FN)
precision = cm[1][1]*100/(cm[1][1]+cm[0][1]) #TP / (TP + FP)
recall = cm[1][1]*100/(cm[1][1]+cm[1][0]) #TP / (TP + FN)
f1_score = 2*precision*recall/(precision+recall) #2 * Precision * Recall / (Precision + Recall)

print('Accuracy', accuracy,'%')
print('Precission', precision,'%')
print('Recall', recall,'%')
print('F1 Score', f1_score,'%')
