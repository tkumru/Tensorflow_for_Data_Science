import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_excel('Tensorflow/bisiklet_fiyatlari.xlsx')

sbn.pairplot(df)
plt.show()

# DISTINGUISH TRAIN AND TEST DATA

# .values convert array to numpy array
y = df["Fiyat"].values  # y is the result variable. So model estimate y. y -> label
x = df[["BisikletOzellik1", "BisikletOzellik2"]].values  # x is the features. Feature means using datas for calculate estimation. x -> feature

# x_train = it is learning data. Model learn with this data.
# x_test = While learning, model understand 'is learning going to true?' data. 
# y_train = it is estimation data for learning.
# y_test = it is controller estimation data 'is true'.
# test_size = %33 of dataframe distinguish for testing
# random_state = it is not important but randomize algorithms

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=36)

#---------------------------------------------------------------------------------

# SCALING

# decrease feature values range from 0 and 1.
# So network can faster. 

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#---------------------------------------------------------------------------------

# CREATE MODEL

model = Sequential() # creation model

model.add(Dense(5, activation="relu"))  # add hidden layer to model with 5 neural network and using ReLU function
model.add(Dense(5, activation="relu"))
model.add(Dense(5, activation="relu"))

model.add(Dense(1))  # output layer with 1 neural network

# mse = mean square error
# optimizer is gradient descent

model.compile(optimizer='rmsprop', loss='mse')

# x_train training with y_train results.
# epochs = how many run model

model.fit(x_train, y_train, epochs=250)  

loss = model.history.history["loss"]  # error margins array
sbn.lineplot(x=range(len(loss)), y=loss)  # graph for error margins
plt.show()

# train loss and test loss should near themself.
# evaulate = Returns the loss value & metrics values for the model in test mode.

train_loss = model.evaluate(x_train, y_train, verbose=0)
test_loss = model.evaluate(x_test, y_test, verbose=0)
print("Train Loss: ", train_loss)
print("Test Loss:", test_loss)

test_estimation = model.predict(x_test)  # estimation of bicycle prices
test_estimation = pd.Series(test_estimation.reshape(330, ))  # 330 is the number of x_test

estimation_df = pd.DataFrame(y_test)
estimation_df = pd.concat([estimation_df, test_estimation], axis=1)  # combine two datafram on axis 1.
estimation_df.columns = ["Real Price", "Estimation Price"]
print(estimation_df)
sbn.scatterplot(x="Real Price", y="Estimation Price", data=estimation_df)
plt.show()

print("Mean Absolute Error: ", mean_absolute_error(estimation_df["Real Price"], estimation_df["Estimation Price"]))
print("Mean Squared Error: ", mean_squared_error(estimation_df["Real Price"], estimation_df["Estimation Price"]))
