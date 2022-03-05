import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from random import randint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

df = pd.read_excel("Tensorflow/maliciousornot.xlsx")

y = df["Type"].values
x = df.drop("Type", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)
print(x_train.shape, "ali")
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape[1], "ali")

model = Sequential()

model.add(Dense(x_train.shape[1], activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(randint(x_train.shape[1]/2, x_train.shape[1]), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(randint(x_train.shape[1]/2, x_train.shape[1]), activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))  # it classifier data so we use sigmoid for output layer

model.compile(loss='binary_crossentropy', optimizer='adam')

# monitor = which data we interested in
# mode = do we looking increasing data or decreasing data?
# patience = if training do not train about 25 epochs
stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x_train, y_train, epochs=700, validation_data=(x_test, y_test), verbose=1, callbacks=[stop])

estimations = model.predict(x_test)
estimations = np.argmax(estimations, axis=1)

print(classification_report(y_test, estimations))
print(confusion_matrix(y_test, estimations))
