# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

### Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray() #Creates multiple columns with 0/1 for countries
X = X[:, 1:] #Remove one column from countries encoding, as it is included already in the rest as info(Dummy variable trap!).

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling -- VERY important always at ANN!!!
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


### Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) #output_dim = (initial_input+final_output)/2

# Adding the second hidden layer -- can remove the input since it knows from the first/previous layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) #RELU/rectifier function is proven to be best for inner layers statistically

# Adding the output layer -- init = 'uniform' guarantees that we are initialising small weights & correctly
# Use 'softmax' as activation if we have more than two categorical outputs at the end. It is like sigmoid
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) #Sigmoid is best for output; gives probability

# Compiling the ANN
# 'adam' is a stochastic gradient descent algorithm.
# loss function: 'binary_crossentropy' is the logarithmic loss function(if more than 2 outcomes, use 'categorical_crossentropy').
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])#metrics->how to evaluate network after each batch

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
classifier.summary()


### Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Evaluate the model
scores = classifier.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))


### Part 4 - Save model to JSON
from keras.models import model_from_json

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
 

### Part 5 - Loading the Model later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Check again the results (Evaluation)
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

probability = loaded_model.predict_proba(X_test)
probability_classes = loaded_model.predict_classes(X_test)

print("probability:", probability)
print("probability classes:", probability_classes)
