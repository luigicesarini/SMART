"""
File name: cnn_msp_tranfer.py
Author: Luigi Cesarini
E-mail: luigi.cesarini@iusspavia.it
Date created: 09 December 2020 
Date last modified: 09 June 2021

####################################################################
PURPOSE:
Perform the task of predicting milk prodiction in the Dominican Republic
using one-dimensional convolutional neural networks trained in Europe
for multi-step ahead predictions.
Also, one-dimensional convolutional neural network models trained and tested 
only on the Dominican Republic data are trained in order to see the improvements
provided by the transferred learning.
"""
# IMPORT MODULE
import os
import time
import datetime
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
from itertools import combinations
from custom_functions import *
from sklearn.metrics import mean_absolute_error,mean_squared_error
#Keras import
from tensorflow import keras
from tensorflow.keras.layers import  Input, Dense, Conv1D, Dropout, Flatten,Multiply, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.backend import square, mean
from tensorflow.python.keras.backend import variable
import warnings
warnings.filterwarnings('ignore')
tf.autograph.set_verbosity(0)
# Set random seed 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
# Store variables for printing purposes
d = datetime.datetime.now()   
date = datetime.datetime.today()
# Features of the model to load
TRAIN = True                # Do we want to train the model on the new country. 
# This is the country I want to make the prediction on with the already trained model.
# It will always be DOM, since we are interested in the prediction only on the Dominican Republic
country = "DOM"            
#List of countries the model has been built for
list_countries = [
    'France',
    'Germany',
    'Italy',
    'DOM'
    ]
# Months ahead that we want the model to predict
forecast_horizon = 12
# The reliable dates of Dominican data
date_start_training    =  "2009-01-01"  #"2012-01-01"
date_start_validation  =  "2014-01-01"  #"2017-01-01"
date_finish_validation =  "2015-01-01"  #"2019-01-01"
date_start_testing     =  "2015-01-01"  #"2019-01-01"
# Initialize two callbacks
# The early stopping is used to stop the training once the minimization of the validation loss reaches a plateau, thus, avoiding overfitting.
# The ReduceLROnPlateau is used to reduce the learning rate of the optimizer when the minimization of the validation loss reaches a plateau
# The patience represents the number of epochs with no improvement after which training will be stopped or learning rate will be reduced.  
patience = 2
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.5,
                                        min_lr=1e-8,
                                        patience=2,
                                        verbose=0)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=patience, verbose=1)

callbacks = [
            callback_early_stopping,
            callback_reduce_lr
            ]
"""
Different lenght of the moving window explored.
For predictions made on the Dominican Republic data,
window larger than 13 months are not allowed, which is
a direct consequence of the few data at disposal and 
the derived split of the data. In fact, the split for 
the dominican data is:
Training:   2009-01-01/2013-12-01
Validation: 2014-01-01/2014-12-01
Testing:    2015-01-01/2015-12-01

Thus, for model tranfer to the DR will be used a maximum lenght of 13 months.
Keep in mind, once the best window for each country is identified,
the loop over 'lenght_window' can be substituted by an if statement that selects
directly the ideal window lenght previously identified.

The same reasoning applies to the exploration of the different combinations
of climatic variables. Once the most important input variables are identified,
their selection can be explicited in the code.
"""
length_window = np.arange(2,14,1)
# How to scale the data. using 'std' will standardized the data  
type_scaling = 'minmax'
#Start the Iteration LOOP
# Initialize a counter to print column names only the first time in the csv
counter_column = 0
for country_model in list_countries:
    for window in length_window:
        #Climate variables used as input
        # - spi: Standard precipitation index 
        # - anomaly_thi: Anomaly of the temperature-humdity index
        # - ndvi: Vegetation index
        training_variables  =  [
                                "spi",
                                "anomaly_thi",
                                "ndv"
                                ] 
        # Iterate over each combination of input data always using the lagged milk as input
        for n_var in range(len(training_variables)+1): 
            combo = []
            for i,names in enumerate(list(combinations(training_variables,n_var))):
                x = list(names)
                x.append('lagged_milk')
                combo.append(x)
            for var in combo:
                # Name of the model used for printing purposes
                if country_model != "DOM":
                    name_method = f'CNN1D_MSP_transfer_{country_model}'
                else:
                    name_method = f'CNN1D_MSP_{country_model}'
                # Load the whole dataset
                data = pd.read_csv(f"MachineLearning/Milk/data/{country}/input_{country}.txt", sep = "\t", header = 0, index_col= 'date')
                # Using the milk of the 1 month lagged milk as input for the model
                data["lagged_milk"] = data.milk.shift(0)
                # Select the months used to train the model. They have to match the period for which the climate data are available
                data = data.loc["2009-01-01":"2015-12-01"] 
                # Store the dates in a variable 
                dates = pd.DataFrame(data.index, columns = ["date"])
                # Reordering of the dataframe
                cols = data.columns.tolist()
                cols.remove('milk')
                cols.append('milk')
                data = data[cols]
                # Features of the model in case. Useful for printing purpose
                type_model = name_method

                # SPLITTING 
                # Set the ranges of validation and testing partition
                start_validation  = dates.date[dates.date == "2014-01-01"].values[0] #dates.date[int(0.7*len(data))]
                finish_validation = dates.date[dates.date == "2014-12-01"].values[0] #dates.date[int(0.7*len(data))]
                start_testing     = dates.date[dates.date == "2015-01-01"].values[0] #dates.date[int(0.9*len(data))]
                finish_testing    = '2015-12-01'

                # Index use to select the warm up period (i.e., the months before the validation and testing period
                # that are equal to the length of the window)
                index_validation  = dates.date[dates.date[dates.date == "2014-01-01"].index-(window+1)].values[0]
                index_testing     = dates.date[dates.date[dates.date == "2015-01-01"].index-(window+1)].values[0]

                #Splitting into the three partition
                training   = data.iloc[np.array(dates.date < start_validation)]
                validation = data.iloc[np.array((dates.date >= index_validation)&(dates.date <= finish_validation))]
                testing    = data.iloc[np.array((dates.date >= index_testing)&(dates.date <= finish_testing))]
                # Split of input and target
                x_train,y_train = training[var], training[["milk"]]
                x_val,y_val     = validation[var],   validation[["milk"]]
                x_test, y_test  = testing[var],  testing[["milk"]]


                # SEPARATE INPUT VARIABLE BY FEATURES AND SCALE
                x_training   = []
                x_validation = []
                x_testing    = []
                for i in var:
                    x_training.append(scale_feature(x_train,x_val,x_test,i,type_scaling)[0])
                    x_validation.append(scale_feature(x_train,x_val,x_test,i,type_scaling)[1])
                    x_testing.append(scale_feature(x_train,x_val,x_test,i,type_scaling)[2])

                x_training      = pd.concat(x_training,axis=1)
                x_validation    = pd.concat(x_validation,axis=1)
                x_testing       = pd.concat(x_testing,axis=1)

                #training, validation and testing partition for the targets
                milk_tr,milk_vl,milk_te = scale_feature(y_train,y_val,y_test,'milk',type_scaling)

                # Splitting into sequences
                # CREATING BATCHES 
                # Training
                x_tr,y_tr   = create_dataset_multistep(x_training,milk_tr,window, forecast_horizon)
                #Validation
                x_vl,y_vl   = create_dataset_multistep(x_validation,milk_vl,window,forecast_horizon)
                #Testing
                x_te,y_te   = create_dataset_multistep(x_testing,milk_te,window,forecast_horizon)

                #Load the trained model or create the new model for the Dominican Republic
                # This step is necessary because no model was trained for DR in the cnn_ssp.py script
                if country_model == 'DOM':
                    model = keras.models.load_model(f'MachineLearning/Milk/trained_model/{country_model}/CNN1D_MSP_{"".join([x[0] for x in var])}_{country_model}_W{window}')
                    TRAIN = False
                else:
                    model = keras.models.load_model(f'MachineLearning/Milk/trained_model/{country_model}/CNN1D_MSP_{"".join([x[0] for x in var])}_{country_model}_W{window}')
                
                # Check summary of loaded model's architecture
                model.summary()
                # Save an image of the network architecture to file    
                tf.keras.utils.plot_model(model,show_shapes = True,show_layer_names = True,to_file=f'MachineLearning/Milk/output/model_{name_method}_{int(time.time())}.png')

                if TRAIN:
                    """
                    Compile the model:
                    Mean absolute error used as loss function during the training.
                    RMSprop use as optimizer to update the weight of the neural network
                    """                

                    model.compile(
                        loss      = "mse",
                        optimizer = keras.optimizers.RMSprop(lr = 1e-4),
                        metrics   = ["mae","mse"],
                    )
                    """ Fitting of the model """
                    history = model.fit(x_tr, y_tr,
                                epochs=500,
                                batch_size = 32, 
                                callbacks = callbacks,
                                verbose = 0,
                                shuffle = False,
                                validation_data = (x_vl,y_vl)
                                )

                # EVALUATION OF MODEL PERFORMANCES
                # At monthly scale
                monthly_mse = []
                monthly_mae = []
                monthly_r2  = []
                for i in range(forecast_horizon):
                    mae_forecast = round(mean_absolute_error(y_te[:,i,0],model.predict(x_te)[:,i]),3)
                    mse_forecast = round(mean_squared_error(y_te[:,i,0],model.predict(x_te)[:,i]),3)
                    r2_forecast  = round(r2_score(y_te[:,i,0],model.predict(x_te)[:,i]),3)
                    monthly_mse.append(mse_forecast)
                    monthly_mae.append(mae_forecast)
                    monthly_r2.append(r2_forecast)

                print(monthly_mse,monthly_mae,monthly_r2)
                print(np.array(monthly_mae))
                # Overall metrics
                mae_tot, mse_tot, r2_tot = [], [], []     
                for i in range(y_te.shape[0]):        
                    mae = mean_absolute_error(y_te[i],model.predict(x_te)[i])    
                    mse = mean_squared_error(y_te[i],model.predict(x_te)[i]) 
                    r2  = r2_score(y_te[i],model.predict(x_te)[i])
                    mae_tot.append(mae)   
                    mse_tot.append(mse) 
                    r2_tot.append(r2)

                overall_mae = np.array(mae_tot).mean()     
                overall_mse = np.array(mse_tot).mean()     
                overall_r2  = np.array(r2_tot).mean()   

                print(f'{country}-{window}-{overall_mae}-{overall_mse}-{overall_r2}')  
                
                #Store metrics for a given configuration in a dictionary
                metrics = {
                    'Country Train':country_model,
                    'Transfer': 'Yes' if country_model != 'DOM' else 'No',
                    'Length of the window':window,
                    'EV Variables': "/".join(x for x in var), # print of environmental variable
                    'Scaling' : type_scaling, #type of scaling adopted
                    'Overall MAE':overall_mae,
                    'Overall MSE':overall_mse,
                    'Overall R^2':overall_r2
                }

                #Create the dataframe containing the dictionary previously created
                df_metrics = pd.DataFrame(metrics, columns=['Country Train', 'Transfer',
                                                            'Length of the window','EV Variables',
                                                            'Scaling',
                                                            'Overall MAE', 'Overall MSE', 'Overall R^2'],
                                                index=[0])
                # Save the metrics to a csv file
                oggi = datetime.datetime.today() #Timestamp for file name
                
                if counter_column == 0:
                    df_metrics.to_csv(f"MachineLearning/Milk/output/metrics_milk_production_{name_method}_{oggi.strftime('%d_%b_%y')}.txt", 
                    quoting=0, index=False, index_label=False, mode='a', header=True, sep='\t')
                else:
                    df_metrics.to_csv(f"MachineLearning/Milk/output/metrics_milk_production_{name_method}_{oggi.strftime('%d_%b_%y')}.txt", 
                    quoting=0, index=False, index_label=False, mode='a', header=False, sep='\t')

                counter_column += 1 #Add element at each configuration trained