# UNDERWATER-MINE-ROCK-IDENTIFICATION-USING-SONAR



## PROJECT OUTLINE:

This mini-project aims to design and develop an underwater mine and rock identification system using sonar technology.
Underwater mines and submerged rocks pose a severe threat to naval vessels, submarines, and commercial ships, making their accurate identification and avoidance crucial.
Sonar technology has emerged as a powerful tool for underwater exploration and threat detection.
The project seeks to provide a cost-effective, reliable, and realtime solution for identifying potential underwater hazards to
enhance safety and security for underwater operations.


## METHODOLOGY:
In this project, we aim to develop an efficient machine-learning model to identify the object in an accurate manner. 
We make use of a dataset,preprocess them, and feed them into a logistic regression model.
Logistic regression is a statistical and machine learning model commonly used for binary classification tasks, where the goal is to
predict one of two possible outcomes, typically denoted as 0 and 1.In this project, we use logistic regression for underwater mine and rock
identification.


## REQUIREMENTS:

* A suitable python environment

* Python packages:
  *  pandas
  *  train_test_split
  *  LogisticRegression
  *  accuracy_score

The above packages can be manually installed using the pip commands as follows:
```
pip install pandas
pip install scikit-learn
pip install tensorflow
```
# PROGRAM:

### IMPORTING NECESSARY PACKAGES:
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```
### Loading the dataset to a pandas DataFrame
```
sonar_data = pd.read_csv('mini_dataset.csv', header=None)
```
### Separating data and labels
```
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```
### Encode labels (R: Rock, M: Mine) to numerical values (0, 1)
```
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
```
### SPLITING DATASET INTO TRAINING SET AND TESTING SET
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
### NEURAL NETWORK MODEL:
```
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),  # Adding dropout for regularization
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
### Train the neural network model
```
model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.1)
train_loss, train_accuracy = model.evaluate(X_train, Y_train)
print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
```
### Evaluate the model on the test data
```
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
```
### Make predictions on the input data
```
input_data = np.array([0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055])
input_data = scaler.transform(np.array(input_data).reshape(1, -1))
prediction = model.predict(input_data)
```
### Convert the prediction to a label ('R' for Rock, 'M' for Mine)
```
predicted_label = 'R' if prediction[0][0] < 0.5 else 'M'
print(f'The object is a {predicted_label}')
```
## FLOW OF THE PROJECT:

1. Load Dataset and Display Dataset
2. Data Preprocessing :
   * Explore Dataset
   * Split Data
3. Neural Network Model:
   * Input Layer
   * Hidden Layer
   * Output Layer
4. Evaluate Model:
   * Calculate Accuracy
5. Output Prediction:
   * Display Prediction



![image](https://github.com/anithapalani2123/UNDERWATER-MINE-ROCK-IDENTIFICATION-USING-SONAR/assets/94184990/040f5080-df0d-477b-b0ba-b5cad021f940)



## OUTPUT:
![image](https://github.com/anithapalani2123/UNDERWATER-MINE-ROCK-IDENTIFICATION-USING-SONAR/assets/94184990/0219ecb4-ac6b-42f2-a2c4-cf923168269f)



## RESULT:
The ultimate goal of the experiment is to create a machine learning model that allows for
accurate predictions of object i.e whether it is underwater mine or rock
This helps us to ensure the safety, security, and efficiency of underwater
operations.















