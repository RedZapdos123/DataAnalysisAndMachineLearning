import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Loading the dataset by taking the file path as input, and adding a technical indicates: RSI, MACD, SMA of last 5 and 10 days,
#Bollinger Bands, ATR and Rate of Price Change.

def DataPreparation(filePath):
    df = pd.read_csv(filePath, parse_dates=['Date'])
    df.sort_values(by='Date', inplace=True)
    df.drop(columns=['Date', 'Adj Close'], inplace=True)
    
    for i in range(1, 6):
        df[f'Open{i}'] = df['Open'].shift(i)
        df[f'High{i}'] = df['High'].shift(i)
        df[f'Low{i}'] = df['Low'].shift(i)
        df[f'Close{i}'] = df['Close'].shift(i)
        df[f'Volume{i}'] = df['Volume'].shift(i)
    
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = ta.rsi(df['Close'])
    
    bbands = ta.bbands(df['Close'])
    df = pd.concat([df, bbands], axis=1)
    
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.roc(df['Close'])
    
    df.dropna(inplace=True)
    return df

#Optimizing the model through baynesian optimization, and fitting the training dataset.
def objective(trial, Xtrain, Xtest, Ytrain, Ytest):
    params ={
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'lambda': trial.suggest_float('lambda', 1, 5),
        'alpha': trial.suggest_float('alpha', 0, 5)
    }
    model = xgb.XGBRegressor(**params, random_state=17, n_jobs=-1)
    model.fit(Xtrain, Ytrain)
    predY = model.predict(Xtest)
    return mean_squared_error(Ytest, predY)

#Splitting the dataset into traing (80%) and testing (20%) dataset, and training the model with optimized XGBoost algorithm.
def trainModel(df):
    X = df.drop(columns=['Close'])
    Y = df['Close']
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=17)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, Xtrain, Xtest, Ytrain, Ytest), n_trials=50)
    
    #Displaying the best hyperarameters for the XGB regressor model, as found by the Optuna Baynesian optimizer.
    print("Best hyperparameters:", study.best_params)
    
    #Fiting and making predictions.
    bestModel = xgb.XGBRegressor(**study.best_params, random_state=17, n_jobs=-1)
    bestModel.fit(Xtrain, Ytrain)
    PredY = bestModel.predict(Xtest)
    
    #Evaluating the model with Mean Squared Error and R2 scores.
    mse = mean_squared_error(Ytest, PredY)
    r2 = r2_score(Ytest, PredY)
    print(f"Mean Squared Error: {mse:.10f}")
    print(f"R² Score: {r2:.10f}")
    
    #Outputing random five day to day predictions.
    for pred, actual in zip(PredY[:5], Ytest[:5]):
        print(f"Predicted: {pred}; Actual: {actual}")
    
    latest_data = X.iloc[-1:].values
    predicted_price = bestModel.predict(latest_data)[0]
    print(f"Predicted price after five trading days: {predicted_price}")
    
    return bestModel, X.columns

#The driver code.
if __name__ == "__main__":
    file_path = input("Enter the stock data CSV file path: ")
    df = DataPreparation(file_path)
    model, feature_names = trainModel(df)
