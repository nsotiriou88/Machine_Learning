# Running my own analysis
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import psycopg2
import sys


def imu_60hz(imu):
    '''imu_60hz is converting the raw imu to 60Hz imu, based on the number of
    imu packets received within each uwb packet. It achieves that by checking
    the previous packet's timestamp and dividing the time with the number of
    the containing imu packets. NOTE: It also removes the first packet, used as
    reference; time zero.'''
    timestamp_first = imu[1][0] - (len(imu[1][1])-1)*(imu[1][0] - imu[0][0])/len(imu[1][1])
    ball_imu = []
    for i in range(1, len(imu)):  # omit first packet
        timestamp_duration = imu[i][0] - imu[i-1][0]
        number_of_imu_samples = len(imu[i][1])
        interim_timestamp = timestamp_duration/number_of_imu_samples
        for j in range(0, len(imu[i][1])):
            ball_imu.append([imu[i][0] - interim_timestamp*(number_of_imu_samples-1-j) - timestamp_first, imu[i][1][j]])
    
    return ball_imu


def imu_60hz_origin(imu):
    '''imu_60hz_origin is converting the raw imu to 60Hz imu, based on the
    number of imu packets received within each uwb packet. It achieves that by
    checking the previous packet's timestamp and dividing the time with the
    number of the containing imu packets. NOTE: It also removes the first
    packet, used as reference; time zero AND uses the first timestamp as TIME
    ORIGIN.'''
    timestamp_first = X[0][1]
    ball_imu = []
    for i in range(1, len(imu)):  # omit first packet
        timestamp_duration = imu[i][0] - imu[i-1][0]
        number_of_imu_samples = len(imu[i][1])
        interim_timestamp = timestamp_duration/number_of_imu_samples
        for j in range(0, len(imu[i][1])):
            ball_imu.append([imu[i][0] - interim_timestamp*(number_of_imu_samples-1-j) - timestamp_first, imu[i][1][j]])
    
    return ball_imu
    
    
def imu_conversion(imu):
    '''imu_conversion is converting the imu values to the normal values scaled
    in G and RPS, for accelerometer and gyroscope respectively.'''
    for i in range(0, len(imu)):
        for j in range(0, len(imu[i][1])):
            for key, val in imu[i][1][j].items():
                if key == 'acc':
                    for keyA, valA in val.items():
                        imu[i][1][j][key][keyA] = valA * 0.732/1000
                elif key == 'gyro':
                    for keyG, valG in val.items():
                        imu[i][1][j][key][keyG] = valG * 70/360/1000
    
    return imu


def imu_expand(imu, Max):
    '''imu_expand is converting the imu from dictionary to simple numpy array
    in the format: [accX, accY, accZ, gyroX, gyroY, gyroZ]. Apply this after
    scaling the IMU data and converting it. USED for CNN training.'''
    temp = np.array([])
    for i in range(0, len(imu)):
        temp = np.append(temp, [[imu[i][1]['acc']['x'], imu[i][1]['acc']['y'], imu[i][1]['acc']['z'], imu[i][1]['gyro']['x'], imu[i][1]['gyro']['y'], imu[i][1]['gyro']['z']]])
    
    temp = np.reshape(temp, (len(imu), 6))
    
    #Try to add padding for the samples to reach Max
    diff = Max - len(imu)
    if diff > 0:        
        for i in range(0, diff//2):
            temp = np.append(temp, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)
            temp = np.insert(temp, 0, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)
        if diff%2 == 1:
            temp = np.append(temp, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)

    return temp
    

#######################
######## START ########
# Importing the dataset
#TRAIN
dataset = pd.read_csv('100passes.csv')
events = pd.read_csv('23Oct_Ealing_SpinPass.csv')
##TEST
#dataset = pd.read_csv('16Oct_Ealing_passes.csv')
#events = pd.read_csv('16Oct_Ealing_passes_time.csv')
##TEST
#dataset = pd.read_csv('10thSep_Ealing_Passes.csv')
#events = pd.read_csv('10thSep_Ealing_Passes_time.csv')

dataset = dataset.sort_values('timestamp')
dataset = dataset.drop(columns=['timelocal', 'force', 'play_id', 'session_id', 'vel_x', 'vel_y', 'vel_z', 'dist'])
X = dataset.iloc[:, :].values

# Collectiong Ball Data only
ball_data = []
for i in range(0, len(X)):
    if X[i][5] == 't' or X[i][5] == True:
        ball_data.append(X[i])

# Get only the imu
imu = []
for i in range(0, len(ball_data)):
    try:
        data = json.loads(ball_data[i][7])
        imu.append([ball_data[i][1], data])
    except:
        continue

# IMU conversion to G and RPS
imu = imu_conversion(imu)

# Extend the 60Hz imu from ball
converted_imu = imu_60hz_origin(imu)

# Events categorisation(passes only, include the previous packet as the zero-time)
Y = events.iloc[:, 0:2].values
    
# Create plotting lists for event packets(including non-ball and non-imu)
passes = []
no_passes = []
for i in range(0, len(Y)):
    passes.append([X[j] for j in range(Y[i][0], Y[i][1])])
    
    # Non-Passes packets
    if i == 0:
        no_passes.append([X[j] for j in range(0, Y[i][0])])
    else: 
        no_passes.append([X[j] for j in range(Y[i-1][1], Y[i][0])])

# Filtering the IMU for ball only, for the event
passes_imu = []
for i in range(0, len(passes)):
    temp = []
    for j in range(0, len(passes[i])):
        if passes[i][j][5] == 't' or passes[i][j][5] == True:
            try:
                data = json.loads(passes[i][j][7])
                temp.append([passes[i][j][1], data])
            except:
                continue
        else:
            continue
    passes_imu.append(temp)

# Filtering the IMU for ball only, for the non-passes
no_passes_imu = []
for i in range(0, len(no_passes)):
    temp = []
    for j in range(0, len(no_passes[i])):
        if no_passes[i][j][5] == 't' or no_passes[i][j][5] == True:
            try:
                data = json.loads(no_passes[i][j][7])
                temp.append([no_passes[i][j][1], data])
            except:
                continue
        else:
            continue
    no_passes_imu.append(temp)

# Signal Conversion (pass)
for i in range(0, len(passes_imu)):
    passes_imu[i] = imu_conversion(passes_imu[i])
    passes_imu[i] = imu_60hz_origin(passes_imu[i])

# Signal Conversion (non-pass)
for i in range(0, len(no_passes_imu)):
    no_passes_imu[i] = imu_conversion(no_passes_imu[i])
    no_passes_imu[i] = imu_60hz_origin(no_passes_imu[i])

# Compare the number of ball_data to ball passes_imu
sum = 0
data_lengths = np.array([])
for i in range(0, len(passes_imu)):
    sum += len(passes_imu[i])
    data_lengths = np.append(data_lengths, len(passes_imu[i]))
    
mean = np.mean(data_lengths)
median = np.median(data_lengths)

# All data from ball (used for detecting passes)
ball_all = imu_expand(converted_imu, 0)


# PLOTTING STUFF
'''
#==================================
# Visualising the Data Acceleration
plt.plot([x[0] for x in converted_imu], [y[1]['acc']['x'] for y in converted_imu], color = 'blue', label='x-acc')
plt.plot([x[0] for x in converted_imu], [y[1]['acc']['y'] for y in converted_imu], color = 'green', label='y-acc')
plt.plot([x[0] for x in converted_imu], [y[1]['acc']['z'] for y in converted_imu], color = 'red', label='z-acc')
plt.axvline(x=50, color = 'yellow')

plt.title('Data Plot')
plt.xlabel('timestamp')
plt.ylabel('Accelerometer')
plt.legend()
plt.show()


# Visualising the Data Gyroscope
plt.plot([x[0] for x in converted_imu], [y[1]['gyro']['x'] for y in converted_imu], color = 'blue', label='x-gyro')
plt.plot([x[0] for x in converted_imu], [y[1]['gyro']['y'] for y in converted_imu], color = 'green', label='y-gyro')
plt.plot([x[0] for x in converted_imu], [y[1]['gyro']['z'] for y in converted_imu], color = 'red', label='z-gyro')

plt.title('Data Plot')
plt.xlabel('timestamp')
plt.ylabel('Gyroscope')
plt.legend()
plt.show()
#==================================

#==================================
# Visualising the Pass Data Acceleration
for i in range(0, len(passes_imu)): 
    plt.plot([x[0] for x in passes_imu[i]], [y[1]['acc']['x'] for y in passes_imu[i]], color = 'blue')
    plt.plot([x[0] for x in passes_imu[i]], [y[1]['acc']['y'] for y in passes_imu[i]], color = 'green')
    plt.plot([x[0] for x in passes_imu[i]], [y[1]['acc']['z'] for y in passes_imu[i]], color = 'red')
plt.axvline(x=50, color = 'yellow')

plt.title('Data Plot')
plt.xlabel('timestamp')
plt.ylabel('Accelerometer')
plt.legend()
plt.show()


# Visualising the Pass Data Gyroscope
for i in range(0, len(passes_imu)): 
    plt.plot([x[0] for x in passes_imu[i]], [y[1]['gyro']['x'] for y in passes_imu[i]], color = 'blue')
    plt.plot([x[0] for x in passes_imu[i]], [y[1]['gyro']['y'] for y in passes_imu[i]], color = 'green')
    plt.plot([x[0] for x in passes_imu[i]], [y[1]['gyro']['z'] for y in passes_imu[i]], color = 'red')
plt.axvline(x=50, color = 'yellow')

plt.title('Data Plot')
plt.xlabel('timestamp')
plt.ylabel('Gyroscope')
plt.legend()
plt.show()
#==================================

#==================================
# Visualising the Non-Pass Data Acceleration
for i in range(0, len(no_passes_imu)): 
    plt.plot([x[0] for x in no_passes_imu[i]], [y[1]['acc']['x'] for y in no_passes_imu[i]], color = 'blue')
    plt.plot([x[0] for x in no_passes_imu[i]], [y[1]['acc']['y'] for y in no_passes_imu[i]], color = 'green')
    plt.plot([x[0] for x in no_passes_imu[i]], [y[1]['acc']['z'] for y in no_passes_imu[i]], color = 'red')
plt.axvline(x=50, color = 'yellow')

plt.title('Data Plot Non-Pass')
plt.xlabel('timestamp')
plt.ylabel('Accelerometer')
plt.legend()
plt.show()


# Visualising the Non-Pass Data Gyroscope
for i in range(0, len(no_passes_imu)): 
    plt.plot([x[0] for x in no_passes_imu[i]], [y[1]['gyro']['x'] for y in no_passes_imu[i]], color = 'blue')
    plt.plot([x[0] for x in no_passes_imu[i]], [y[1]['gyro']['y'] for y in no_passes_imu[i]], color = 'green')
    plt.plot([x[0] for x in no_passes_imu[i]], [y[1]['gyro']['z'] for y in no_passes_imu[i]], color = 'red')
plt.axvline(x=50, color = 'yellow')

plt.title('Data Plot Non-Pass')
plt.xlabel('timestamp')
plt.ylabel('Gyroscope')
plt.legend()
plt.show()
#==================================
#==================================
'''

# CNN Implementation

### Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense
from keras.layers import Reshape, GlobalAveragePooling1D, Dropout

# Initialising the CNN
classifier = Sequential()

# Reshape the Layer for accepting data - no need; doing it manually.
#classifier.add(Reshape((81, 6), input_shape=(486,))) # 6x81

# Step 1 - Convolution
classifier.add(Convolution1D(100, 10, input_shape = (81, 6), activation = 'relu'))
#classifier.add(Convolution1D(100, 10, activation = 'relu'))

# Step 2 - Pooling (sliding the feature detector; removes computational complexity by reducing feature map size)
classifier.add(MaxPooling1D(2))

# Adding a second convolutional layer - or less better, add another fully connected layer.
classifier.add(Convolution1D(160, 10, activation = 'relu'))
#classifier.add(Convolution1D(160, 10, activation = 'relu'))
classifier.add(MaxPooling1D(2))
#classifier.add(GlobalAveragePooling1D()) # Do not use with flattening

# Step 3 - Flattening
classifier.add(Flatten())
#classifier.add(Dropout(0.5)) # Prevent overfitting; do not use with flattening

# Step 4 - Full connection
classifier.add(Dense(output_dim = 486, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


### Part 2 - Fitting the CNN to the IMU data
# Data preparation
max_sample_length = 81
# PASS
trainX = []
for i in range(0, len(passes_imu)):
    trainX.append([imu_expand(passes_imu[i], max_sample_length)])
# NON-PASS
no_pass_trainX = []
for i in range(0, len(no_passes_imu)):
    no_pass_trainX.append([imu_expand(no_passes_imu[i], max_sample_length)])

# Create a sliding window with step 20 and length 81 samples for big samples of non passes.
no_passes_slide = []
for interval in no_pass_trainX:
    samples = len(interval[0])
    if samples > max_sample_length:
        pointer = 0
        while pointer+max_sample_length != samples:
            no_passes_slide.append([interval[0][i] for i in range(pointer, pointer+max_sample_length)])
            if pointer+20+max_sample_length < samples:
                pointer += 20
            elif pointer+20+max_sample_length == samples:
                pointer += 20
                no_passes_slide.append([interval[0][i] for i in range(pointer, pointer+max_sample_length)])
            else:
                break
#                pointer += pointer+20+max_sample_length-samples-3
#                no_passes_slide.append([interval[0][i] for i in range(pointer, pointer+max_sample_length)])
no_passes_slide = np.array(no_passes_slide)
y_no_pass = np.zeros((len(no_passes_slide), 1))

# Convert list of list of numpy arrays to numpy array
trainX_np = np.array([])
for i in range(0, len(trainX)):
    trainX_np = np.append(trainX_np, trainX[i][0])
trainX_np = np.reshape(trainX_np, (len(trainX), max_sample_length, 6))
y = np.ones((len(trainX), 1))

# Concatenate the passes and non-passes data
X_train_all = np.concatenate((trainX_np, no_passes_slide), axis=0)
y_all = np.concatenate((y, y_no_pass), axis=0)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_all, test_size = 0.1, random_state = 0)

# Do I need to normalise the data(maybe between different sensors; acc & gyro
# and not x, y & z)???

# Fit Generator model
#from keras.preprocessing.sequence import TimeseriesGenerator
#sampling_data_train = TimeseriesGenerator(X_train, X_train, 5)
#sampling_data_test = TimeseriesGenerator(X_test, X_test, 5)

#classifier.fit_generator(sampling_data_train,
#                         samples_per_epoch = 90,
#                         nb_epoch = 25,
#                         validation_data = sampling_data_test,
#                         nb_val_samples = 10)

history = classifier.fit(X_train_all,
                         y_all,
                         batch_size=20,
                         epochs=20, #optimise number of epochs for overfitting
#                         validation_split=0.1,
                         verbose=1)

# Classifier summary
classifier.summary()
loss, accuracy = classifier.evaluate(X_test, y_test)
print('loss: ', loss, '\naccuracy: ', accuracy)
#print(classifier.predict(X_test), y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
cm = confusion_matrix(y_test, y_pred)

#============
#============

# Testing sliding window for detection.
# Create a sliding window for the data in order to scan for pass detection.
test_sliding = []
total_frequency_sliding = []
sliding_range = [i for i in range(5, 81, 5)]
sliding_range.insert(0, 1)
for sliding in sliding_range:
    ball_slide = []
    frequency_sliding = np.zeros(len(ball_all))
    pointer = 0
    while pointer+max_sample_length != len(ball_all):
        ball_slide.append([ball_all[i] for i in range(pointer, pointer+max_sample_length)])
        if pointer+sliding+max_sample_length < len(ball_all):
            pointer += sliding
        elif pointer+sliding+max_sample_length == len(ball_all):
            pointer += sliding
            ball_slide.append([ball_all[i] for i in range(pointer, pointer+max_sample_length)])
        else:
            break
    ball_slide = np.array(ball_slide)
    if sliding == 1:
        ball_slide_step1 = ball_slide
    
    # Count the number of passes based on a sliding feeding window of max_sample_length.
    pass_pred = classifier.predict(ball_slide)
    
    pass_count = 0
    for j in range(0, len(pass_pred)):
        if pass_pred[j] > 0.5:
            pass_count += 1
            for k in range(j*sliding, j*sliding+max_sample_length):
                frequency_sliding[k] += 1
    total_frequency_sliding.append(frequency_sliding)
    
    test_sliding.append([sliding, pass_count])
    print("Passes detected on dataset:", pass_count, "Stepping:", sliding)

# Create histogram of how many packets were flagged as passes.
# Normalise
count_pass_from_probability = []
for j in range(0, len(total_frequency_sliding)):
    min = np.min(total_frequency_sliding[j])
    max = np.max(total_frequency_sliding[j])
    counter = 0
    for k in range(0, len(total_frequency_sliding[j])):
        total_frequency_sliding[j][k] = (total_frequency_sliding[j][k]-min)/(max-min)
        if total_frequency_sliding[j][k] > 0.6:
            counter += 1
    count_pass_from_probability.append(counter)
    
# Calculate peaks by moving detector mask 75% of the length of the mask every
# time it detects a pass, or 100% or 120%.
pointer = 0
new_pass_count = 0
while pointer+max_sample_length <= len(ball_slide_step1):
    if classifier.predict(np.array([ball_slide_step1[pointer]])) >= 0.5:
        new_pass_count += 1
        pointer += int(1.2*max_sample_length)
#        pointer += int(0.75*max_sample_length)
#        pointer += max_sample_length
    else:
        pointer += 1

# Plot results
'''
plt.plot(total_frequency_sliding[4], 'g-', label='Probability(centre) stepping 25')
plt.axvline(x=50, color = 'yellow')

plt.title('Pass Probability Plot')
plt.xlabel('samples')
plt.ylabel('Probability (centre)')
plt.legend()
plt.show()
'''

plt.plot(total_frequency_sliding[1], 'b-', label='Probability(centre) stepping 60')
plt.axvline(x=50, color = 'yellow')

plt.title('Pass Probability Plot')
plt.xlabel('samples')
plt.ylabel('Probability (centre)')
plt.legend()
plt.show()
'''

plt.plot(total_frequency_sliding[11], 'r-', label='Probability(centre) stepping 65')
plt.axvline(x=50, color = 'yellow')

plt.title('Pass Probability Plot')
plt.xlabel('samples')
plt.ylabel('Probability (centre)')
plt.legend()
plt.show()
'''
#============
'''
#Peak Detector
import peakutils
from peakutils.plot import plot as pplot

x = [j for j in range(len(total_frequency_sliding[0]))]
plt.figure(figsize=(10,6))
plt.plot(x, total_frequency_sliding[0])

indexes = peakutils.indexes(total_frequency_sliding[0], thres=0.01/np.max(total_frequency_sliding[0]), min_dist=60)
x = np.array(x)
plt.figure(figsize=(10,6))
pplot(x, total_frequency_sliding[0], indexes)
'''

sys.exit()
#============
#============
# Save model
classifier.save('partly_trained_1263-69.h5')
#classifier.save('partly_trained_69.h5')

# Load Model
from keras.models import load_model
classifier = load_model('partly_trained_1263-69.h5')
#classifier = load_model('partly_trained_69.h5')

#============
#============

# Exporting CSV data from the server
conn = psycopg2.connect(host="192.168.1.150", database="metrics_server", user="metrics_server", password="Sp0rtableM3ss13r106", port="5432")
curs = conn.cursor()
query = "SELECT * FROM processed_packets WHERE session_id=1163 ORDER BY timestamp"
outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
with open('output.csv', 'w') as file:
    curs.copy_expert(outputquery, file)

#curs.execute("SELECT * FROM processed_packets WHERE session_id=1163 ORDER BY timestamp")
#query = curs.fetchone()


#============
