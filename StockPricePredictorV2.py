#This program uses the eXtreme Gradient descent Boosting with robust scaling algorithm to predict the stock prices,
#from it's five days of opening and closing prices, trading volumes and through technical indicators like
#Smiple Moving Averages of last five and ten days and volatility, momentum and price changes of the last day.
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

#Loading and the preparing the dataset for processing. The file path is an user input.
def DataPreparation(filePath):
    df = pd.read_csv(filePath, parse_dates=['Date'])
    df.sort_values(by='Date', inplace=True)
    df.drop(columns=['Date', 'Adj Close'], inplace=True)

    #Using the last five days' data as features for the model.
    for i in range(1, 6):
        df[f'Open{i}'] = df['Open'].shift(i)
        df[f'High{i}'] = df['High'].shift(i)
        df[f'Low{i}'] = df['Low'].shift(i)
        df[f'Close{i}'] = df['Close'].shift(i)
        df[f'Volume{i}'] = df['Volume'].shift(i)
    
    #Using 5 days' and 10 days' simple moving averages (SMA), price change percentage, stock momentum and volatility as features
    df['PriceChange'] = df['Close'].pct_change(1)
    df['Volatility'] = (df['High'] - df['Low']) / df['Close']
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    
    #Dropping the rows with null values.
    df.dropna(inplace=True)
    
    return df

#Training the model using the extreme gradient descent boosting with robust scaler algorithm.
def trainModel(df):
    X = df.drop(columns=['Close'])
    Y = df['Close']
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 17)
    
    #Scaling the features for training the model.
    scaler = RobustScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    #Making the XGBoost model with parameters.
    XGBmodel = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators = 200,
        max_depth = 5,
        learning_rate = 0.1,
        subsample = 0.9,
        colsample_bytree = 0.9,
        random_state = 17,
        n_jobs = -1
    )
    
    #Fitting and predicting using the XGBoost model.
    XGBmodel.fit(Xtrain, Ytrain)
    PredY = XGBmodel.predict(Xtest)

    #Evaluating the model.
    mse = mean_squared_error(Ytest, PredY)
    r2 = r2_score(Ytest, PredY)
    print(f"Mean Squared Error: {mse:.10f}")
    print(f"R² Score: {r2:.10f}")
    
    #Printing the first five predictions.
    for pred, actual in zip(PredY[:5], Ytest[:5]):
        print(f"Predicted: {pred}, Actual: {actual}")
    
    return XGBmodel, scaler, X.columns

#Calling the functions in the driver code.
if __name__ == "__main__":
    file_path = input("Enter the stock data CSV file path: ")
    df = DataPreparation(file_path)
    model, scaler, feature_names = trainModel(df)

