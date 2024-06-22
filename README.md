# Stock Price Prediction using LSTM

## Description
This project implements a Long Short-Term Memory (LSTM) network to predict the stock prices of Microsoft Corporation (MSFT). It includes two Python scripts: `stock-prediction.py`, which trains the LSTM model and tests its accuracy using historical stock price data, and `predictions.py`, which trains the model and predicts future stock prices for the next 7 days.

## Getting Started

### Prerequisites
- Python 3.x
- Pip package manager

### Installation

1. Clone the repository to your local machine.

2. Install the required packages:
    pip install -r requirements.txt
    *note that depending on your development environment and other factors, you might have to install different versions of these using pip3 install packagename

## Running the Code

To run the stock prediction: python3 stock-prediction.py

This script will train a LSTM model using the historical stock data provided and test the model's performance on the test data. The output will be a graph showing the true stock values versus the model's predictions.

To run future stock price prediction: python3 predictions.py

This script will train a LSTM model using all the historical stock data provided, and then predict the stock prices for the next 7 days. The output will be a graph showing the predicted stock price changes for the upcoming week.

## Understanding the Scripts

- `stock-prediction.py`: Trains a LSTM model on a provided dataset, splitting it into training and testing subsets to evaluate the model's performance. The script outputs a visualization of the model's predictions against the actual stock prices from the test set.

- `predictions.py`: Trains a LSTM model on a provided dataset, using all the data for training, to forecast future stock prices. It takes the last 60 days of closing prices to predict the next 7 days' prices and outputs a graph showing the predicted future prices.
