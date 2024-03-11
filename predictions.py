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
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

#Get the Dataset
df = pd.read_csv('MSFT (1).csv', na_values=['null'], index_col='Date', parse_dates=True)
df.head()

# Print the shape of DataFrame and Check for Null Values
print("DataFrame Shape:", df.shape)
print("Null Values Present:", df.isnull().values.any())

#Plot the True Adj Close Value
df["Adj Close"].plot()
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
# Adjusting LSTM layer and adding Dropout
lstm = Sequential()
lstm.add(LSTM(64, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dropout(0.2))  # Dropout for regularization
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Function to decay the learning rate
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Learning rate scheduler
lr_scheduler = LearningRateScheduler(scheduler)

# Train the LSTM model with early stopping and learning rate scheduler
history = lstm.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=8, 
    verbose=1, 
    shuffle=False,
    validation_split=0.2,  # Adjust this if you have a separate validation set
    callbacks=[early_stopping, lr_scheduler]
)


history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction
y_pred= lstm.predict(X_test)

test_dates = df.index[len(train_index): (len(train_index)+len(test_index))]

lstm.save('my_model.keras')

# Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)

# Assuming your LSTM model is already trained and is named 'lstm'
# And assuming your model takes the last 60 days of data to make a prediction
# (this window size can be different based on how you trained your model)

# Prepare the last 60 days of data as input for the predictions
last_60_days = feature_transform[-60:]

# Reshape the data to fit the model input shape
last_60_days_scaled = scaler.transform(last_60_days)
X_test = last_60_days_scaled.reshape((1, 60, len(features)))

# Make predictions for the next 7 days
predicted_prices = []
for i in range(7):  # for one week
    # Predict the next day price
    predicted_price = lstm.predict(X_test)
    # Store the prediction
    predicted_prices.append(predicted_price[0, 0])
    # Add the prediction to the end of the window and remove the first element
    # This reshaping creates a 2D array with one column (since our prediction is one feature)
    new_row_for_X_test = np.zeros((1, len(features)))
    new_row_for_X_test[0, 0] = predicted_price[0, 0]  # Assumes 'Adj Close' is at index 0
    X_test = np.append(X_test[:, 1:, :], new_row_for_X_test[:, np.newaxis, :], axis=1)

# To inverse transform the predicted prices, create an array with the same number of columns as the original data
# Initialize an array of zeros with the required shape
predicted_prices_array = np.zeros((len(predicted_prices), len(features)))
# Fill in the 'Adj Close' predictions into the first column (or the appropriate column index)
predicted_prices_array[:, 0] = predicted_prices

predicted_prices_scaled = np.zeros((len(predicted_prices), len(features)))
predicted_prices_scaled[:, 0] = predicted_prices
predicted_prices_transformed = scaler.inverse_transform(predicted_prices_scaled)[:, 0]

for i, price in enumerate(predicted_prices_transformed, start=1):
    print(f"Day {i}: Predicted Price: {price}")


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-Squared: {r2}')


#Plot the predictions
plt.figure(figsize=(10, 5))
predicted_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
plt.plot(predicted_dates, predicted_prices, label='LSTM Predicted Price Change for Next Week')
plt.title('Future Stock Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.legend()
plt.show()