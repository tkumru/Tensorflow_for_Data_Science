import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_excel('Tensorflow/merc.xlsx')

print(df.describe())
print()
print(df.isnull().sum())  # is there any empty value
print()

# we analyse there what distribution of car selled which price. Examp: The mostly range sell car price is 25000 pound
sbn.distplot(df["price"])  # distribution graph for price
plt.show()

# we analyse there how many car selled which year
sbn.countplot(df["year"])
plt.show()

# we analyse there relationship between price column and other columns
correlation = df.corr()["price"].sort_values()
print(correlation)

# we write mileage because mileage is most effective column on price
sbn.scatterplot(x='mileage', y='price', data=df)
plt.show()

# we remove highest values from dataframe.
# we decided removed value number to calculating %1 of dataframe. It is totaly optional.
# we removed highest because when we analyse distribution graph very low number of distribition highest values.
# we removed because %1 values of dataframe can break neural network training.
sorted_df = df.sort_values("price", ascending=False)  # Sort values highest to lowest.
df = sorted_df.iloc[int(len(df) * 0.01): ]

print(df.describe())

# we see that 1970's mean 25k around. It is so damn. Decided to 1970's car.
print(df.groupby("year").mean()["price"])
df = df[df.year != 1970]
df = df.drop("transmission", axis=1)  # because it is string value
# WE CAN NOT DELETE TRANSMISSION. WE CAN CATEGORIZE STRING VALUES.#

sbn.distplot(df["price"])  # distribution graph for price
plt.show()

# Creating Model

y = df["price"].values
x = df.drop("price", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = Sequential()

model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=300, epochs=200)

loss_data = pd.DataFrame(model.history.history)
loss_data.plot()
plt.show()

estimation_series_x = model.predict(x_test)
real_series_y = y_test
plt.scatter(y_test, estimation_series_x)
plt.show()

print("Mean Absolute Error: ", mean_absolute_error(y_test, estimation_series_x))
print("Mean Squared Error: ", mean_squared_error(y_test, estimation_series_x))
