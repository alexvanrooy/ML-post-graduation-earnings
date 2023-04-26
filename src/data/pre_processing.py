#IMPORTS
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from os.path import exists
from zipfile import ZipFile
import os

class CleanDataset:
    '''
    Class that has methods for cleaning and processing the given dataset
    '''
    def __init__(self, input_path, output_path):
        '''
        Takes input file path and loads the dataset.

                Parameters:
                        input_path (str): the file path to the dataset
                        output_path (str): the file path to where to store the cleaned dataset

                Returns:
                        None
        '''
        self.input_path = input_path
        self.output_path = output_path
        self.raw_data = None
        self.cleaned_data = None

        return

    def loadData(self):
        '''
        Loads the raw data into a DataFrame

                Parameters:
                    None

                Returns:
                        success (int): if the data is loaded then return 1, otherwise 0
        '''

        success = 0
        try:
            file_exists = exists(self.input_path)
            if(not file_exists):
               print("Unzipping Dataset...")
               with ZipFile("..\data\external\Most-Recent-Cohorts-Institution.zip", 'r') as zObject:
                    zObject.extractall("..\data\external")
                    print("SUCCESS: Dataset unzipped!")

            print("Attempting to load the dataset...")
            self.raw_data = pd.read_csv(self.input_path, low_memory=False)
            success = 1
            print("SUCCESS: Dataset loaded.")

        except Exception as e:
            print("ERROR: Could not load dataset.")
            print("\nMake sure you are running the program from within the src folder.")
            print('\nCURRENT DIRECTORY: ', os.getcwd(),"\n")
            print(e)
        
        return success


    def clean(self, nan_ratio = 0.4):
        '''
        Takes the raw dataset and cleans it by removing rows and columns that don't have enough data, and replacing missing values.

                Parameters:
                        nan_ratio (float): the threshold of NaN values that determines if a row/column gets removed.

                Returns:
                        success (int): if the data is loaded then return 1, otherwise 0
        '''
        success = 0
        try:
            print("Starting data cleaning process...")
            self.cleaned_data = self.raw_data.copy()
            
            #get all the columns that are not numeric
            categorical_columns = [col for col in self.cleaned_data.columns if self.cleaned_data[col].dtype == 'object']

            #convert the categorical columns to numeric so values that are not numbers are turned to NaN (i.e. PrivacySupressed values)
            count = 0
            col_len = len(categorical_columns)
            
            for col in categorical_columns:
                count += 1
                self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='coerce')
                sys.stdout.write(f"Converting columns to numeric: {count}/{col_len} \r")
                sys.stdout.flush()

            sys.stdout.write('\n')
            
            #collect all the columns that are above the NaN threshold and drop them from the dataset
            nanRatioOver50 = [col for col in self.cleaned_data.columns if (self.cleaned_data[col].isna().sum() / self.cleaned_data.shape[0]) > nan_ratio]
            print("Removing bad columns...")
            self.cleaned_data = self.cleaned_data.drop(nanRatioOver50, axis=1)

            #get all the rows that are missing lots of data and remove them from the dataset
            bad_rows = [index for index, row in self.cleaned_data.iterrows() if (row.isna().sum() / self.cleaned_data.shape[1]) > nan_ratio]
            print("Removing bad rows...")
            self.cleaned_data = self.cleaned_data.drop(index = bad_rows, axis = 0)

            #replace all the remaining NaN values with the mean of the column
            print("Replacing missing values...")
            self.cleaned_data = self.cleaned_data.fillna(self.cleaned_data.mean())
            
            print("SUCCESS: Data was cleaned successfully.")
            success = 1
        
        except Exception as e:
            print("ERROR: Failed to clean data.")
            print(e)

        return success

    def outputClean(self):
        '''
        Outputs the cleaned dataset to the specified output path in .csv format

                Parameters:
                        None

                Returns:
                        success (int): returns 1 if ouput was successful, 0 otherwise.
        '''
        success = 0

        try:
            print("Attempting to output cleaned dataset...")
            self.cleaned_data.to_csv(self.output_path)
            success = 1
            print(f"SUCCESS: Cleaned data output to {self.output_path}")
        except Exception as e:
            print("ERROR: Unable to export cleaned data as csv.")
            print(e)
        
        return success
    
class SplitData:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None 
        self.X_test = None
        self.y_train = None 
        self.y_test = None

        return
    
    def loadData(self):
        '''
        Loads the X and y dataset.

                Parameters:
                    None

                Returns:
                        success (int): if the data is loaded then return 1, otherwise 0
        '''

        success = 0
        print("Attempting to load the processed dataset...")
        try:
            self.X = pd.read_csv('..\data\interim\X.csv', index_col= 0)
            self.y = pd.read_csv('..\data\interim\y.csv', index_col= 0)

            success = 1
            print("SUCCESS: Dataset loaded.")

        except Exception as e:
            print("ERORR: Could not load dataset.")
            print(e)
        
        return success


    def splitData(self, test_size = 0.3, random_state = 1):
         
        print("Attempting to split the data...")
        success = 0
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
            print("SUCCESS: Data was split into a train set and test set.")
            
            print('Samples in training set: ', self.X_train.shape[0])
            print('Samples in testing set: ', self.X_test.shape[0])
            success = 1

        except Exception as e:
            print("ERROR: Something went wrong.")
            print(e)
        
        return success

    def outputSplitData(self):

        success = 0

        print("Attempting to export training and testing sets...")
        try:
            pd.DataFrame(self.X_train).to_csv('..\data\processed\X_train.csv')
            pd.DataFrame(self.X_test).to_csv('..\data\processed\X_test.csv')
            pd.DataFrame(self.y_train).to_csv('..\data\processed\y_train.csv')
            pd.DataFrame(self.y_test).to_csv('..\data\processed\y_test.csv')
            success = 1
        except Exception as e:
            print("Something went wrong.")
            print(e)
        return success