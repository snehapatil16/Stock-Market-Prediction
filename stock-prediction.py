#Importing the Libraries
import pandas as pd
import numpy as np
#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as Ks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import plot_model

#Get the Dataset
df = pd.read_csv('MSFT (1).csv', na_values=['null'], index_col='Date', parse_dates=True)
df.head()

# Print the shape of DataFrame and Check for Null Values
print("DataFrame Shape:", df.shape)
print("Null Values Present:", df.isnull().values.any())

#Plot the True Adj Close Value
df["Adj Close"].plot(title="Adj Close Values for MSFT Stock")
# plt.show()

#Set Target Variable
output_var = pd.DataFrame(df["Adj Close"])
#Selecting the Features
features = ['Open', 'High', 'Low', 'Volume']

#Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()

#Splitting to Training set and Test set
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

# Print the sizes of the last train and test sets
print("Last Train Set Size:", len(train_index))
print("Last Test Set Size:", len(test_index))

#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes=True, show_layer_names=True)

history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction
y_pred= lstm.predict(X_test)

test_dates = df.index[len(train_index): (len(train_index)+len(test_index))]

lstm.save('my_model.keras')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(test_dates, y_test, label='True Value')  # Plotting y_test with test_dates
plt.plot(test_dates, y_pred, label='LSTM Prediction')  # Plotting y_pred with test_dates
plt.title('Prediction by LSTM')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()
