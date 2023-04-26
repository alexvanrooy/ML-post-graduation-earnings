import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import numpy as np

class Visualizer():
    ''' 
    Class that generates visuals based on model results and data exploration
    '''
    def __init__(self, models):
        self.models = models
        self.y_test = None
        self.X = None
        self.y = None

    def loadData(self):
        ''' 
        Loads the test data for the target variable as well as the complete dataset for features and target
        '''
        success = 0
        print("Attempting to load data...")
        try:
            self.y_test = pd.read_csv('..\data\processed\y_test.csv', index_col=0)
            self.X = pd.read_csv('..\data\interim\X.csv', index_col=0)
            self.y = pd.read_csv('..\data\interim\y.csv', index_col=0)
            print("SUCCESS: Testing data loaded.")
            success = 1
        except Exception as e:
            print("ERROR: Unable to load data.")
            print(e)
        return success 

    def compareFit(self):
        '''
        Compares the models fit to the true fit of the data
        '''

        try:
            #building the dataframe for plotting
            target_row = pd.Series(self.y_test.to_numpy().ravel(), name = "TRUE")
            data = pd.DataFrame(target_row)

            for model in self.models:
                name = model.name
                model_pred = model.y_pred
                model_row = pd.Series(model_pred, name = name)
                data = pd.concat([data,model_row], axis = 1)


            data = data.sort_values(by = 'TRUE')

            for col in data.columns:
                data[col] = data[col].rolling(50).mean()

            data = data.dropna().reset_index(drop = True)
            
            ax = data.plot.line()

            ax.set_title("Predicted VS True Mean Income 6 Years Post-Graduation")
            ax.set_xlabel('Sample Number')
            ax.set_ylabel('Mean Income')
            
            plt.savefig("..\\reports\\figures\model_fit.png")

            plt.show()


        except Exception as e:
            print("ERROR: Something went wrong.")
            print(e)
        
        return
    
    def metricsTable(self):
        ''' 
        Compare the r-square, rmse, and mae for each model
        '''

        if(len(self.models) == 0):
            print("No trained models to compare.")
            return
        try:
            data = pd.DataFrame(columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error', 'R-Squared'])

            for model in self.models:
                row = pd.Series({'Model' : model.name, 'Root Mean Squared Error': model.rmse, 'Mean Absolute Error':model.mae, 'R-Squared':model.r_squared})
                data = pd.concat([data, row.to_frame().T], ignore_index=True)
            
            
            ax = plt.subplot(111, frame_on = False)
            ax.axis('tight')
            ax.axis('off')
            ax.set_title("Comparing Model Metrics")
            ax.xaxis.set_visible(False) # hide the x axis
            ax.yaxis.set_visible(False) # hide the y axis
            
            table(ax, data, loc='center') # where df is your data frame
            
            plt.savefig("..\\reports\\figures\model_metrics.png")
            plt.show()
            

        except Exception as e:
            print("ERROR: Something went wrong.")
            print(e)

    def accuracyBarPlot(self):
        ''' 
        Shows the accuracy of each model at different sizes
        '''
        if(len(self.models) == 0):
            print("No trained models to compare.")
            return

        try:
            data = pd.DataFrame({"Interval" : ['< 10,000', '< 8,000', '< 5,000', '< 2,000']})


            for model in self.models:
                diff = np.absolute(self.y_test.to_numpy().ravel() - model.y_pred.ravel())
                diff10000 = np.count_nonzero(diff < 10000)
                diff8000 = np.count_nonzero(diff < 8000)
                diff5000 = np.count_nonzero(diff < 5000)
                diff2000 = np.count_nonzero(diff < 2000)

                acc10000 = diff10000/self.y_test.shape[0]
                acc8000 = diff8000/self.y_test.shape[0]
                acc5000 = diff5000/self.y_test.shape[0]
                acc2000 = diff2000/self.y_test.shape[0]

                new_row = pd.DataFrame({model.name : [ acc10000, acc8000, acc5000, acc2000]})
                data = data.join(new_row)

            ax = data.plot(x = 'Interval', kind = 'bar', stacked = False, rot = 0, title = "Accuracy of Models Within Different Intervals", ylabel= "Accuracy")
            plt.savefig("..\\reports\\figures\model_accuracy.png")
            plt.show()

        except Exception as e:
            print("ERROR: Something went wrong.")
            print(e)
        
        return

    def correlations(self):
        ''' 
        Examines the correlations of a few select features
        '''
        try:

            #how family income can effect withdrawal from school
            x_feature = 'WDRAW_ORIG_YR2_RT'
            y_feature = 'MD_FAMINC'
            d = {'Withdrawal Rate' : self.X[x_feature], 'Median Family Income' : self.X[y_feature]}
            df = pd.DataFrame(d).sort_values(by = 'Withdrawal Rate')

            x_feature = 'Withdrawal Rate'
            y_feature = 'Median Family Income'

            fit = np.polyfit(df[x_feature], df[y_feature], 2)
            a = fit[0]
            b = fit[1]
            c = fit[2]
            fit_equation = a * np.square(df[x_feature]) + b * df[x_feature] + c

            plt.scatter(x = df[x_feature], y = df[y_feature])

            plt.plot(df[x_feature], fit_equation, color = 'red')
            plt.xlabel(x_feature)
            plt.ylabel(y_feature) 
            plt.title("Withdrawal Rate VS Median Family Income")

            plt.savefig("..\\reports\\figures\correlation_withdraw_vs_income.png")
            plt.show()

            #how school tution revenue can affect completion rate at a school
            x_feature = 'COMP_ORIG_YR4_RT'
            y_feature = 'TUITFTE'

            d = {'Completion Rate' : self.X[x_feature], 'Net Tuition Revenue' : self.X[y_feature]}
            df = pd.DataFrame(d).sort_values(by = 'Completion Rate')

            x_feature = 'Completion Rate'
            y_feature = 'Net Tuition Revenue'

            fit = np.polyfit(df[x_feature], df[y_feature], 2)
            a = fit[0]
            b = fit[1]
            c = fit[2]
            fit_equation = a * np.square(df[x_feature]) + b * df[x_feature] + c

            plt.scatter(x = df[x_feature], y = df[y_feature])

            plt.plot(df[x_feature], fit_equation, color = 'red')
            plt.xlabel(x_feature)
            plt.ylabel(y_feature) 
            plt.title("Completion Rate VS Net Tuition Revenue")
            plt.savefig("..\\reports\\figures\correlation_completion_vs_tuition.png")
            plt.show()

            #shows that a big part of predicting
            x_feature = 'FAMINC'
            y_feature = 'MN_EARN_WNE_P6'

            d = {'Family Income' : self.X[x_feature], 'Mean Earnings 6 Years After Graduation' : self.y.to_numpy().ravel()}
            df = pd.DataFrame(d).sort_values(by = 'Family Income')

            x_feature = 'Family Income'
            y_feature = 'Mean Earnings 6 Years After Graduation'

            fit = np.polyfit(df[x_feature], df[y_feature], 2)
            a = fit[0]
            b = fit[1]
            c = fit[2]
            fit_equation = a * np.square(df[x_feature]) + b * df[x_feature] + c

            plt.scatter(x = df[x_feature], y = df[y_feature])

            plt.plot(df[x_feature], fit_equation, color = 'red')
            plt.xlabel(x_feature)
            plt.ylabel(y_feature) 
            plt.title("Family Income Rate VS Mean Earnings 6 Years After Graduation")
            plt.savefig("..\\reports\\figures\correlation_family_income_vs_mean_earnings.png")
            plt.show()

        except Exception as e:
            print("ERROR: Something went wrong.")
            print(e)


