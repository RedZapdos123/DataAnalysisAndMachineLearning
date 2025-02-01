#A program to manually implement multiple linear regression with standard scaling model 
# to predict stock prices of a stock from it's last day's opening, highest and lowest prices and it's trading volume.
#The dataset should have columns in this format: Date, Open, High, Low, Close, Adjusted Close, Volume.

import numpy as np
import pandas as pd

#Loading and performing preprocessing on the dataset.
def loadData(filePath):
    try:
        data = pd.read_csv(filePath)
        print("File loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filePath}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: File is not in the correct format.")
        return None

#Taking the file path as user input.
filePath = input("Input the datasets' file path: ")

data = loadData(filePath)

if data is not None:
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
else:
    exit("Data loading failed. Exited.")

#Adding previous day's Closing price as column by shifting the rows of 'Close' columns downwards by one.
data['PreviousClose'] = data['Close'].shift(1)
data.dropna(inplace = True)

#Standardizing the feature data columns.
features = ['Open', 'High', 'Low', 'Volume', 'PreviousClose']
meanValues = data[features].mean()
stdValues = data[features].std()
X = (data[features] - meanValues) / stdValues

#Keeping Y (the closing prices) in original scale.
Y = data['Close'].values

#Adding the bias term.
X = np.hstack((np.ones((X.shape[0], 1)), X))

#Shuffling the data for better training and testing sets. 
def shuffleData(X, Y):
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    return X[indexes], Y[indexes]

X, Y = shuffleData(X, Y)

#Splitting data into training and testing sets, in 80:20 ratio.
def dataSplit(X, Y, testSize = 0.2):
    trainSize = int(len(X) * (1 - testSize))
    Xtrain, Xtest = X[:trainSize], X[trainSize:]
    Ytrain, Ytest = Y[:trainSize], Y[trainSize:]
    return Xtrain, Xtest, Ytrain, Ytest

Xtrain, Xtest, Ytrain, Ytest = dataSplit(X, Y, testSize=0.2)

#Initializing the weights.
np.random.seed(69)
w = np.random.randn(Xtrain.shape[1])

#The Mean Squared Error function.
def MSE(X, Y, w):
    m = X.shape[0]
    PredY = X.dot(w)
    return (1/(2*m)) * np.sum((PredY - Y)**2)

#The Gradient Descent function.
def gradientDescent(X, Y, w, learningRate, epochs):
    m = X.shape[0]
    errors = []

    for epoch in range(epochs):
        PredY = X.dot(w)
        dw = (1/m) * X.T.dot(PredY - Y)
        w -= learningRate*dw
        error = MSE(X, Y, w)
        errors.append(error)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Error: {error}")

    return w, errors

#Training the model with the parameters.
learningRate = 0.005
epochs = 5000
w, errors = gradientDescent(Xtrain, Ytrain, w, learningRate, epochs)

#The Prediction function.
def predict(X, w):
    return X.dot(w)

PredY = predict(Xtest, w)

#Evaluating the performance.
MSEValue = np.mean((PredY - Ytest)**2)
print(f"Mean Squared Error: {MSEValue}")

#Using C-efficient of Determination (R2) method.
R2Value = 1 - ( np.sum((Ytest-PredY)**2)/np.sum((Ytest - np.mean(Ytest))**2))
print(f"Coefficient of Determination (R2): {R2Value}")

#Displaying the first five predictions against actual values.
for i in range(min(5, len(PredY))):
    print(f"Predicted: {PredY[i]}; Actual: {Ytest[i]}")
