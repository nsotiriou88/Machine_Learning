# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras


### Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# First we have the number and size of feature detectors; input_shape is 2D for black/white and 3D for coloured.
# Theano has opposite order for input_shape parameters ==> (3, 64, 64).
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling (sliding the feature detector; removes computational complexity by reducing feature map size)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer - or less better, add another fully connected layer.
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection - common practice 128
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #for non-binary, 'softmax' activation is required

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #'categorical_crossentropy' for non-binary


### Part 2 - Fitting the CNN to the images - avoid overfitting by doing image augmentation process

from keras.preprocessing.image import ImageDataGenerator

#transformations to be applied for making images different each time they are fed to the network
#=== Augmentation objects for applying them to our images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
#======

print("Training Set Preprocessing")
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), #dimensions of the input_shape
                                                 batch_size = 32, #frequency of weight adjustments
                                                 class_mode = 'binary') #or 'categorical'

print("Test Set Preprocessing")
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

# Classifier summary
classifier.summary()
loss, accuracy = classifier.evaluate(test_set)
print('loss: ', loss, '\naccuracy: ', accuracy)


### Part 3 - Save model to JSON
from keras.models import model_from_json

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
 

### Part 4 - Loading the Model later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
json_file.close()

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.summary()

## Check again the results (Evaluation)
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(test_set, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# Prediction testing
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('prediction1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image) # <==== loaded model here
probability = loaded_model.predict_proba(test_image)
probability_classes = loaded_model.predict_classes(test_image)
print(training_set.class_indices) # <===== Check if the predictions are set correctly

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("result:", result)
print("predictions:", prediction)
print("probability:", probability)
print("probability classes:", probability_classes)
