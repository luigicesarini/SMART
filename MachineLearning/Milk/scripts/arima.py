"""
File name: arima.py
Author: Luigi Cesarini
E-mail: luigi.cesarini@iusspavia.it
Date created: 21 November 2020 
Date last modified: 07 June 2021

####################################################################
PURPOSE:
Perform the task of predicting milk prodiction using a univariate ARIMA model
from the statsmodel module:
https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
"""
#IMPORT MODULES
import os 
import time
import datetime
from datetime import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')

#Set working directory to the parent of the script folder
#os.chdir('..')
print(os.getcwd())

# Set random seed for reproduciblity
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
#Store date for printing purposes
date = datetime.today().strftime('%d_%b_%y')
################################
#DEFINITION OF CUSTOM FUNCTION
################################
"""
Define a function that prints the performances of the models:
**args
- type_model: name identifying the tyoe of model
- MSE,MAE, and R^2 for the training and testing partition
- to_txt: boolean value that allows the saving of the metrics to a txt 
"""
def print_performances(type_model,
                       mse_train, 
                       mse_test,
                       mae_train, 
                       mae_test,
                       r2_train, 
                       r2_test,
                       to_txt = True
                       ):
    prepare_print = {
        "Model"     :   type_model,
        "date"      :   datetime.today().ctime,
        "mse_train" :   mse_train, 
        "mse_test"  :   mse_test, 
        "mae_train" :   mae_train,
        "mae_test"  :   mae_test,
        "r2_tr"     :   r2_train,
        "r2_te"     :   r2_test,
    }
    if to_txt:
        pd.DataFrame(prepare_print, index = [0]).to_csv(f"MachineLearning/Milk/output/metrics_milk_production_{date}.csv",
                        quoting=0, index=False, index_label=False, mode='a', header=False, sep='\t')
    else:
        print(prepare_print)


# 1) ARIMA MODEL
countries = [
    'Italy',
    'France',
    'Germany',
    'DOM'
]

scaling = input("Type of scaling to use?\n(Write minmax for normalization\nWrite std for standardization)\n")

# Create the ARIMA model for each country interested
for country in countries:

    data = pd.read_csv(f"MachineLearning/Milk/data/{country}/input_{country}.txt", sep = "\t", header = 0, index_col=0, parse_dates=[0] )
    
    #Store the milk time series in a variable
    train_series = data.milk
    #Plot of the original time series
    train_series.plot()
    plt.ylabel('Milk Production in million of liters')
    plt.title(f"{country}'s original time series")
    plt.show()
    # Remove name index column name
    train_series.index.name = None
    #  Split data in training and testing partition
    x_train = train_series.loc['1982-01-01':'2015-12-01']
    x_test  = train_series.loc['2016-01-01':'2019-12-01']

    # Data scaling
    if scaling == 'std':
        #Retrieve mean and standard deviation
        mean_tr, std_tr = x_train.mean(),x_train.std()
        # Standardization of the two partition
        x_train = (x_train-mean_tr)/std_tr
        x_test  = (x_test-mean_tr)/std_tr
    else:
        #Retrieve minimum and maximum 
        min_tr, max_tr = x_train.min(),x_train.max()
        # Normalization of the two partition
        x_train = (x_train-min_tr)/(max_tr-min_tr)
        x_test  = (x_test-min_tr)/(max_tr-min_tr)



    # Fit a model with initial guess of the arima parameters
    model = ARIMA(x_train, order=(12,1,6))
    model_fit = model.fit()
    # Print summary of the statistics of the fitted model
    print(model_fit.summary())

    # Plot the residuals
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals to check the kind of distribution and the value around which the mean of the residual is 
    residuals.plot(kind='kde')
    plt.show()
    # Summary stats of residuals
    print(residuals.describe())

    
    # Plot of autocorrelation and partial autocorrelation
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(train_series.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(train_series, lags=40, ax=ax2)
    plt.show()

    # Print the split of training/testing data
    print(x_train.shape[0]/data.shape[0],x_test.shape[0]/data.shape[0])


    #QQ-plot of the residual
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    fig = qqplot(model_fit.resid, line='q', ax=ax, fit=True)
    plt.show()

    """
    Below a routine aimed at findind the best parameters p,d,q for this case.
    The for loop iterates between all the possible combinations of the 3 parameters,
    fits the ARIMA mode, when possible, and prints the performances to a txt file.
    """

    import itertools
    p = range(2,25)
    q = range(1,25)
    d = range(1,3)
    pdq = list(itertools.product(p,d,q))

    for iter in range(len(pdq)):
        if iter % 50 == 0:
            print(iter)
        try:
            param = (pdq[iter][0],pdq[iter][1],pdq[iter][2]) 
            model = ARIMA(x_train, order=param).fit()

            predict_milk = model.forecast(steps =len(x_test))

            print_performances(
                    type_model = f"ARIMA {param} - {country}",
                    mae_train  = round(sum(abs(model.predict()-x_train))/len(x_train),3),
                    mae_test   = round(sum(abs(predict_milk-x_test))/len(x_test),3),
                    mse_train  = round(sum((model.predict()-x_train)**2)/len(x_train),3),
                    mse_test   = round(sum((predict_milk-x_test)**2)/len(x_test),3),
                    r2_train   = round(r2_score(x_train,model.predict()),3),
                    r2_test    = round(r2_score(x_test,predict_milk),3),
                    to_txt     = True
                    )
        except:
            continue

