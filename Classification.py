import Clustering
import Prediction

import numpy as np
from sklearn import preprocessing
from keras.wrappers.scikit_learn import  KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


epochs = 15
batch_size = 100

latent_dim = 50
classes = Clustering.unique_labels
input_dim = Prediction.num_input_features
def baseline_model():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(None, input_dim)))
    model.add(Dropout(0.1))
    model.add(LSTM(latent_dim))
    model.add(Dropout(0.1))
    model.add(Dense(len(classes), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


x_train = np.asarray(Prediction.series,dtype=Prediction.series[0].dtype)
y_train = Clustering.labels

encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
encoder_Y = encoder.transform(y_train)

dummy_y = np_utils.to_categorical(encoder_Y)

estimator = KerasClassifier(build_fn=baseline_model,epochs=epochs, batch_size=batch_size)
X_train, X_test, Y_train, Y_test = train_test_split(x_train, dummy_y, test_size=0.2)
estimator.fit(X_train, Y_train)

