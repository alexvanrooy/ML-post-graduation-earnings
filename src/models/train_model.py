#IMPORTS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from  sklearn.linear_model import Lasso


def loadTrainingData():
    '''
    Loads the data that will be used to train the models
    '''
    X_train = None
    y_train = None
    success = 0

    print("Attempting to load training data...")
    try:
        X_train = pd.read_csv('..\data\processed\X_train.csv', index_col=0)
        y_train = pd.read_csv('..\data\processed\y_train.csv', index_col=0)
        print("SUCCESS: Training data loaded.")
        success = 1
    except Exception as e: 
        print("ERROR: Unable to load training data.")
        print(e)
    
    return X_train, y_train, success

class SVM:
    '''
    Class that holds the necessary methods for creating and training an SVM regression model
    '''

    def __init__(self, X_train, y_train, random_state = 1):

        self.name = 'SVM'
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.model = None
        self.random_state = random_state
        

        return
    
    def standardizeData(self):
        '''
        Standardizes the data to be used for SVM training
        '''
        
        scaler = StandardScaler()
        X_train_SVM = scaler.fit_transform(self.X_train, self.y_train)
        self.X_train = X_train_SVM
        
        return
    
    def train(self, option = 'bag'):
        success = 0
        if(option == 'bag'):
            try:

                print("Creating SVM Model...")
                self.model = BaggingRegressor(SVR(kernel='rbf', C= 185000, gamma = 0.0126), n_estimators=90, max_features=0.65, random_state=self.random_state)
                print("Model created.")

                print("Training SVM Model with Bagging (Can take up to 10 minutes to complete)...")
                self.model.fit(self.X_train, self.y_train.to_numpy().ravel())
                print("SVM finished training.")
                success = 1
                
            except Exception as e:
                print("ERROR: Unable to train SVM model.")
                print(e)
        else:
            try:
                print("Creating SVM Model...")
                self.model = SVR(kernel='rbf', C= 180000, gamma = 0.013)
                print("Model created.")

                print("Training SVM Model...")
                self.model.fit(self.X_train, self.y_train.to_numpy().ravel())
                print("SVM finished training.")
                success = 1
            except Exception as e:
                print("ERROR: Unable to train SVM model.")
                print(e)

        return success
    
class DecisionTree:
    '''
    Class that holds the necessary methods for creating an training a Decision Tree regression model.
    '''
    def __init__(self, X_train, y_train, random_state = 1):

        self.name = 'Decision Tree'
        self.X_train = X_train.copy()
        self.y_train = y_train.copy().to_numpy().ravel()
        self.model = None
        self.random_state = random_state
        
        return

    def train(self):
        success = 0
        try:
            #parameters for decision tree model
            max_depth = 20
            min_samples_split = 5
            max_features = 'sqrt'
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features, random_state=self.random_state)
            #parameters for bagging model
            n_estimators = 90
            max_samples = 0.95
            max_features = 1.0

            print("Creating Decision Tree Model with Bagging...")
            bag = BaggingRegressor(model, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, random_state=self.random_state)
            self.model = bag
            print("Model Created.")

            print("Training Decision Tree Model...")
            self.model.fit(self.X_train, self.y_train)
            print("Decision Tree model finished training.")
            success = 1

        except Exception as e:
            print("ERROR: Unable to train DecisionTree Model")
            print(e)
        
        return success

class KNN:
    ''' 
    Class that holds the necessary methods for creating and training a KNN regression model
    '''
    def __init__(self, X_train, y_train, random_state = 1):

        self.name = 'KNN'
        self.X_train = X_train.copy()
        self.y_train = y_train.copy().to_numpy().ravel()
        self.model = None
        self.random_state = random_state

        return
    
    def scaleData(self):
        ''' 
        Scale the training data so it can be used with KNN
        '''
        self.X_train = self.X_train/self.X_train.max()
        return
    
    def train(self):
        success = 0
        try:
            
            print("Creating KNN Model with Bagging...")
            knn = KNeighborsRegressor(n_neighbors=2)
            self.model = BaggingRegressor(knn, n_estimators=30, max_features=0.55, random_state=self.random_state)
            print("Model Created.")

            print("Training the KNN Model...")
            self.model.fit(self.X_train, self.y_train)
            print("KNN Model finished training.")
            success = 1

        except Exception as e:
            print("ERROR: Unable to train KNN Model.")
            print(e)
        
        return success
    
class LassoModel:
    '''
    Class that contains all the necessary methods for creating and training a Lasso regression model
    '''
    def __init__(self, X_train, y_train, random_state = 1):

        self.name = 'Lasso'
        self.X_train = X_train.copy()
        self.y_train = y_train.copy().to_numpy().ravel()
        self.model = None
        self.random_state = random_state
        return
    
    def scaleData(self):
        '''
        Scale the training data so it can be used with Lasso
        '''
        self.X_train = self.X_train/self.X_train.max()
        return

    def train(self):
        success = 0

        try:

            print("Creating Lasso Model...")
            self.model = Lasso(alpha = 1, random_state=self.random_state)
            print("Model Created.")

            print("Training the Lasso Model...")
            self.model.fit(self.X_train, self.y_train)
            print("Lasso Model finished training.")
            success = 1
        except Exception as e: 
            print("ERROR: Unable to train Lasso Model.")
            print(e)
        
        return success