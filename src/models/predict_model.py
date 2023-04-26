#IMPORTS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

def loadTestingData():
    '''
    Load the data to use for testing
    '''

    X_test = None
    y_test = None
    success = 0

    print("Attempting to load testing data...")
    try:
        X_test = pd.read_csv('..\data\processed\X_test.csv', index_col=0)
        y_test = pd.read_csv('..\data\processed\y_test.csv', index_col=0)
        print("SUCCESS: Testing data loaded.")
        success = 1
    except Exception as e: 
        print("ERROR: Unable to load test data.")
        print(e)
    
    return X_test, y_test, success

def comparePredictors(predictors):
    '''
    Generates a command line table to compare the models
    '''

    print('Model Name    | R-Squared | Root Mean Squared Error | Mean Absolute Error |')
    print('--------------|-----------|-------------------------|---------------------|')
    best_model = predictors[0]
    for predictor in predictors:
        if(predictor.r_squared > best_model.r_squared):
            best_model = predictor
        print(f"{predictor.name:14}|{predictor.r_squared:11.2f}|{predictor.rmse:25.2f}|{predictor.mae:21.2f}|")
    
    print("")
    print(f"Best Model: {best_model.name}")
    
    return best_model

class modelPredictor:
    '''
    Class that contains the methods for preparing testing data as well as testing models
    '''

    def __init__(self,model, X_test, y_test):

        self.name = model.name
        self.model = model.model
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.y_pred = None
        self.r_squared = None
        self.mae = None
        self.rmse = None

        return

    def standardizeData(self):
        
        scaler = StandardScaler()
        X_test_SVM = scaler.fit_transform(self.X_test, self.y_test)
        self.X_test = X_test_SVM
        
        return
    
    def scaleData(self):
        self.X_test = self.X_test/self.X_test.max()
        return
    
    def predict(self):
        success = 0

        try:
            print("Attempting to predict results using test data...")
            if(self.name == 'SVM'):
                print("(This will take a few minutes)")
                
            self.y_pred = self.model.predict(self.X_test)
            print("Prediction complete.")
            success = 1
        except Exception as e:
            print("ERROR: Unable to make prediction.")
            print(e)
        
        return success
    
    def generateStats(self):
        success = 0

        try:
            print("Computing R-Squared...")
            self.r_squared = self.model.score(self.X_test, self.y_test)
            
            print("Computing Root Mean Squared Error...")
            self.rmse = mean_squared_error(y_true=self.y_test, y_pred=self.y_pred, squared=False)

            print("Computing Mean Absolute Error...")
            self.mae = mean_absolute_error(y_true=self.y_test, y_pred=self.y_pred)
            
            print(f"\nResults for {self.name}")
            print("--------------------------")
            print(f"R-Squared = {self.r_squared}")
            print(f"Root Mean Squared Error = {self.rmse}")
            print(f"Mean Absolute Error = {self.mae}")
            
            
            success = 1
        except Exception as e: 
            print("ERROR: Unable to compute metrics for model.")
            print(e)

        return success
