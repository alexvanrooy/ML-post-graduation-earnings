'''
The main python script for the project
'''

#IMPORTS
from data.pre_processing import CleanDataset, SplitData
from features.build_features import select_features
from models.train_model import loadTrainingData, SVM, DecisionTree, KNN, LassoModel
from models.predict_model import loadTestingData, modelPredictor, comparePredictors
from visualization.visualize import Visualizer
import os

#GLOBAL VARIABLES
WELCOME_MESSAGE = '''Estimating Earnings After Graduation from Post-Secondary Institutions	
by: Alex Van Rooy
---------------------------------------------------------------------
Please Select an Option Below:

1. Cleaning Data
2. Extracting Features
3. Split data into Training and Test Sets
4. Select Models for Training
5. Make Predictions on Test Set
6. Generate Visuals
7. Exit
'''

#Trained Models
modelSVM = ...
modelDecisionTree = ...
modelKNN = ...
modelLasso = ...

#Model status (0 = not trained, 1 = trained)
svmStatus = 0
decisionTreeStatus = 0
knnStatus = 0
lassoStatus = 0

#Model Predictors
svmPredictor = None
decisionTreePredictor = None
knnPredictor = None
lassoPredictor = None

best_model = None


#METHODS
def prepare():
    ''' 
    Method to perform the data cleaning process.
    '''
    success = 0
    os.system('cls')
    print("Cleaning The Data")
    print("------------------")

    input_path = "..\data\external\Most-Recent-Cohorts-Institution.csv"
    output_path = "..\data\interim\Most-Recent-Cohorts-Institution_CLEANED.csv"

    clean_data = CleanDataset(input_path=input_path, output_path=output_path)

    #Load the dataset
    if(not clean_data.loadData()):
        input("\nPress ENTER to return home.")
        return success

    #Clean the dataset
    if(not clean_data.clean()):
        input("\nPress ENTER to return home.")
        return success

    #Export the dataset
    if(not clean_data.outputClean()):
        input("\nPress ENTER to return home.")
        return success

    print("Data successfully cleaned!")
    success = 1
    input("\nPress ENTER to return home.")
    return success

def features():
    ''' 
    Method to turn cleaned data into predictors and target features.
    '''
    os.system('cls')
    print("Selecting Features")
    print("------------------")
    success = select_features()
    
    if(success == 1):
        print("Successfully extracted features!")
    
    input("\nPress ENTER to return home.")
    return success

def splitData():
    '''
    Runs the scripts to split the processed data into training and testing set
    '''
    
    os.system('cls')
    print("Split Data into Training & Testing Sets")
    print("---------------------------------------")

    split = SplitData()
    success = 0

    if(not split.loadData()):
        input("\nPress ENTER to return home.")
        return success
    
    if(not split.splitData()):
        input("\nPress ENTER to return home.")
        return success
    
    if(not split.outputSplitData()):
        input("\nPress ENTER to return home.")
        return success
    
    print("Testing set and Training set created successfully!")
    success = 1
    input("\nPress ENTER to return home.")
    
    return success

def trainModels():
    ''' 
    Asks the user which model they wish to train and runs the necessary scripts
    '''
    os.system('cls')
    print("Training Models")
    print("---------------")

    #Load training data
    X_train, y_train, success = loadTrainingData()
    print("")

    if(success == 0):
        input("\nPress ENTER to return home.")
        return   

    while(1):
        #print model selection text
        print('''Select the Models you want to train:

1. SVM
2. Decision Tree
3. KNN
4. Lasso
5. Exit
''')
        print("")
        user_input = input("Select Option: ")
        print('')
        if(user_input == '5'):
            break
        #Create and train SVM model
        elif(user_input == '1'):
            global modelSVM
            modelSVM = SVM(X_train, y_train)                        #initialize model object
            modelSVM.standardizeData()                              #standardize the training data
            global svmStatus
            svmStatus = modelSVM.train()                            #fit the model to data
            input("\nPress ENTER to continue.")

        #Create and train Decision Tree model
        elif(user_input == '2'):
            global modelDecisionTree
            modelDecisionTree = DecisionTree(X_train, y_train)
            global decisionTreeStatus
            decisionTreeStatus = modelDecisionTree.train()
            input("\nPress ENTER to continue.")

        #Create and train KNN model
        elif(user_input == '3'):
            global modelKNN
            modelKNN = KNN(X_train, y_train)
            modelKNN.scaleData()
            global knnStatus
            knnStatus = modelKNN.train()
            input("\nPress ENTER to continue.")

        #Create and train Lasso model
        elif(user_input == '4'):
            global modelLasso
            modelLasso = LassoModel(X_train, y_train)
            modelLasso.scaleData()
            global lassoStatus
            lassoStatus = modelLasso.train()
            input("\nPress ENTER to continue.")
            
        os.system('cls')
        print('''Training Models
---------------''')
    return

def predictModels():
    '''
    Ask the user which model they want to test and runs the necessary scripts
    '''
    
    os.system('cls')
    print("Predicting with Models")
    print("----------------------")
    
    #Load training data
    X_test, y_test, success = loadTestingData()
    print("")

    if(success == 0):
        input("\nPress ENTER to return home.")
        return
    
    while(1):
        #print model prediction test
        print('''Select the Model you want to use for predicting:

1. SVM
2. Decision Tree
3. KNN
4. Lasso
5. Compare Results
6. Exit
''')
        print("")
        user_input = input("Select Option: ")
        print('')

        #Exit prediction menu
        if(user_input == '6'):
            break
        
        #test the SVM model
        elif(user_input == '1'):
            if (svmStatus == 0):
                print("ERROR: You must train the model before it can be used for predictions.")
            else:
                global svmPredictor
                svmPredictor = modelPredictor(model = modelSVM, X_test=X_test, y_test=y_test)
                svmPredictor.standardizeData()
                svmPredictor.predict()
                svmPredictor.generateStats()
        
        #test the decision tree model
        elif(user_input == '2'):
            if(decisionTreeStatus == 0):
                print("ERROR: You must train the model before it can be used for predictions.")
            else:
                global decisionTreePredictor
                decisionTreePredictor = modelPredictor(model = modelDecisionTree, X_test=X_test, y_test=y_test)
                decisionTreePredictor.predict()
                decisionTreePredictor.generateStats()
        
        #test the knn model
        elif(user_input == '3'):
            if(knnStatus == 0):
                print("ERROR: You must train the model before it can be used for predictions.")
            else:
                global knnPredictor
                knnPredictor = modelPredictor(model = modelKNN, X_test=X_test, y_test=y_test)
                knnPredictor.scaleData()
                knnPredictor.predict()
                knnPredictor.generateStats()

        #test the lasso model
        elif(user_input == '4'):
            if(lassoStatus == 0):
                print("ERROR: You must train the model before it can be used for predictions.")
            else:
                global lassoPredictor
                lassoPredictor = modelPredictor(model = modelLasso, X_test=X_test, y_test=y_test)
                lassoPredictor.scaleData()
                lassoPredictor.predict()
                lassoPredictor.generateStats()
        
        #Compare all the tested models
        elif(user_input == '5'):
            predictors = []
            if(svmPredictor is not None):
                predictors.append(svmPredictor)
            
            if(decisionTreePredictor is not None):
                predictors.append(decisionTreePredictor)
            
            if(knnPredictor is not None):
                predictors.append(knnPredictor)
            
            if(lassoPredictor is not None):
                predictors.append(lassoPredictor)
            
            if(len(predictors) > 0):
                global best_model
                best_model = comparePredictors(predictors)
            else:
                print("ERROR: No predictors to compare yet.")
            

        input("\nPress ENTER to continue.")
        os.system('cls')
        print("Predicting with Models")
        print("----------------------")
    return

def generateVisuals():
    ''' 
    Method that will run the scripts for generating visualizations
    '''

    os.system('cls')
    print("Generate Visualizers")
    print("--------------------")

    #Get the models that have been tested
    models = []
    if(svmPredictor is not None):
        models.append(svmPredictor)
    
    if(decisionTreePredictor is not None):
        models.append(decisionTreePredictor)
    
    if(knnPredictor is not None):
        models.append(knnPredictor)
    
    if(lassoPredictor is not None):
        models.append(lassoPredictor)

    #create visualizer object
    visuals = Visualizer(models)
    
    #Run the visualizations
    if(visuals.loadData() == 0):
        input("Press ENTER to return home.")
    else:
        print("Generating Visuals...")
        visuals.compareFit()
        visuals.metricsTable()
        visuals.accuracyBarPlot()
        visuals.correlations()    

    input("Press ENTER to continue.")
    return

check_src = os.getcwd().split(sep='\\')[-1]
#Main Loop
while(1):
    os.system('cls')

    #print the choices for the user
    print(WELCOME_MESSAGE)

    #check to make sure user is in the src folder when running the programs
    if(check_src != 'src'):
        print("**WARNING: Please run the main.py file from inside the src folder otherwise the program will not work.**\n")

    #print the status of the models
    print("Model Status:")
    if(svmStatus == 1):
        print("           SVM = Trained")
    else:
        print("           SVM = Untrained")
    
    if(decisionTreeStatus == 1):
        print(" Decision Tree = Trained")
    else:
        print(" Decision Tree = Untrained")
    
    if(knnStatus == 1):
        print("           KNN = Trained")
    else:
        print("           KNN = Untrained")
    
    if(lassoStatus == 1):
        print("         Lasso = Trained")
    else:
        print("         Lasso = Untrained")

    print('')

    if(best_model is not None):
        print('')
        print(f"Best Model: {best_model.name}\n")


    user_input = input("Select Option: ")

    #Exit the program
    if(user_input == '7'):
        break
    
    #Clean the dataset
    elif(user_input == '1'):
        prepare()

    #Extract features
    elif(user_input == '2'):
        features()

    #Split the data
    elif(user_input == '3'):
        splitData()
    
    #train models
    elif(user_input == '4'):
        trainModels()

    #use models to predict test data
    elif(user_input == '5'):
        predictModels()

    #generate visualizations using the data and results
    elif(user_input == '6'):
        generateVisuals()