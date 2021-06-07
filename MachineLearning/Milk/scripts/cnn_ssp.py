"""
File name: cnn_ssp.py
Author: Luigi Cesarini
E-mail: luigi.cesarini@iusspavia.it
Date created: 27 November 2020 
Date last modified: 07 June 2021

####################################################################
PURPOSE:
Perform the task of predicting milk prodiction using a one-dimensional
convolutional neural network implemented through tensorflow.
The models trained will be used for single-step ahead predictions

"""
# IMPORT MODULE
import os 
import time
import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from itertools import combinations
#TF imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
# import custom functions
from custom_functions import plot_time_series,scale_feature,create_dataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
# Store variables for printing purposes
d = datetime.datetime.now()   
date = datetime.datetime.today()
name_method = 'CNN1D'

"""
Define a function that prints the performances of the models:
**args
- type_model: name identifying the tyoe of model
- MSE,MAE, and R^2 for the training, validation and testings partition
- to_txt: boolean value that allows the saving of the metrics to a txt """
def print_performances(type_model,
                       mse_train, mse_val, mse_test,
                       mae_train, mae_val, mae_test,
                       r2_train, r2_val, r2_test,
                       name_method,
                       to_txt = True
                       ):
    prepare_print = {
        "Model"     :   f'{type_model}_{country}',
        "date"      :   d.ctime(),
        "mse_train" :   mse_train, 
        "mse_val"   :   mse_val, 
        "mse_test"  :   mse_test, 
        "mae_train" :   mae_train,
        "mae_val"   :   mae_val,
        "mae_test"  :   mae_test,
        "r2_tr"     :   r2_train,
        "r2_vl"     :   r2_val,
        "r2_te"     :   r2_test,

    }
    if to_txt:
        pd.DataFrame(prepare_print, index = [0]).to_csv(f"MachineLearning/Milk/output/metrics_milk_production_{name_method}_{date.today().strftime('%d_%m_%Y')}.txt",
                    quoting=0, index=False, index_label=False, mode='a', header=False, sep='\t')
    else:
        print(prepare_print)
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
#List of countries the model has been built for
countries = [
    'France',
    'Germany',
    'Italy',
    ]
# Different lenght of the moving window explored
length_window = np.arange(2,24,1)
# How to scale the data. using 'std' will standardized the data  
type_scaling = 'minmax'
# 4b) CNN 1D 6 variables:
for country in countries:
    for window in length_window:
        #Climate variables used as input
        # - spi: Standard precipitation index 
        # - anomaly_thi: Anomaly of the temperature-humdity index
        # - ndvi: Vegetation index
        training_variables  =  ["spi",
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
                # Load the whole dataset
                data = pd.read_csv(f"MachineLearning/Milk/data/{country}/input_{country}.txt", sep = "\t", header = 0, index_col= 'date')
                # Using the milk of the 1 month lagged milk as input for the model
                data["lagged_milk"] = data.milk.shift(0)
                # Select the months used to train the model. They have to match the period for which the climate data are available
                data = data.loc["1982-02-01":"2019-11-01"] # !! REMEMBER: Dropping NA we loose the last month
                # Store the dates in a variable 
                dates = pd.DataFrame(data.index, columns = ["date"])
                # Reordering of the dataframe
                cols = data.columns.tolist()
                cols.remove('milk')
                cols.append('milk')
                data = data[cols]
                # Features of the model in case. Useful for printing purpose
                type_model = f"CNN1 {type_scaling} {[x for x in var]} W:{window}"

                # SPLITTING 
                # Set the ranges of validation and testing partition
                start_validation = dates.date[dates.date == "2009-01-01"].values[0] #dates.date[int(0.7*len(data))]
                start_testing    = dates.date[dates.date == "2016-01-01"].values[0] #dates.date[int(0.9*len(data))]
                finish_testing   = dates.date[-1:].values[0]

                # Index use to select the warm up period (i.e., the months before the validation and testing period
                # that are equal to the length of the window)
                index_validation  = dates.date[dates.date[dates.date == "2009-01-01"].index-window].values[0]
                index_testing     = dates.date[dates.date[dates.date == "2016-01-01"].index-window].values[0]
                #Splitting into the three partition
                training   = data.iloc[np.array(dates.date < start_validation)]
                validation = data.iloc[np.array((dates.date >= index_validation)&(dates.date <= start_testing))]
                testing    = data.iloc[np.array(dates.date >= index_testing)]
                # Split of input a target
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
                x_tr,y_tr   = create_dataset(x_training,milk_tr,window)
                #Validation
                x_vl,y_vl   = create_dataset(x_validation,milk_vl,window)
                #Testing
                x_te,y_te   = create_dataset(x_testing,milk_te,window)

                # Build the model with the functional API.
                # Using only one LSTM layer
                # Clear previous possible model still in memory
                tf.keras.backend.clear_session()

                # CNN1D Model
                inputs       = keras.layers.Input(shape = (x_tr.shape[1],x_tr.shape[2])) 
                conv1_1      = keras.layers.Conv1D(filters = 128, kernel_size = 2,#int(window/2),
                                                     activation = 'relu', padding = 'same', name = "CNN_1_1")(inputs)
                flatten_1    = keras.layers.Flatten(name = "Flat_1")(conv1_1)
                dense_1      = keras.layers.Dense(100, name = "Dense_1")(flatten_1)
                drop_1       = keras.layers.Dropout(0.2, name = "Drop_2")(dense_1)
                outputs = keras.layers.Dense(1, name = "Predictions")(drop_1)

                model = keras.Model(inputs=inputs, outputs = outputs)
                # Check summary of model's architecture
                print(model.summary())
                # Save an image of the network architecture to file    
                tf.keras.utils.plot_model(model,show_shapes = True,show_layer_names = True,to_file="model_{}.png".format(int(time.time())))
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
                            #validation_split = 0.5,
                            batch_size = 64, #int(0.1*x_tr.shape[0]),
                            callbacks = callbacks,
                            verbose = 0,
                            shuffle = False,
                            validation_data = (x_vl,y_vl)
                            )

                # Save the trained model. This way the model can be loaded at a later stage 
                # and be used to make prediction on the Dominican Republic or any other country. 
                model.save(f'MachineLearning/Milk/trained_model/{country}/CNN1D_{"".join([x[0] for x in var])}_{country}_W{window}')
                # Print the performances of the model to disk
                print_performances(type_model,
                                    mae_train   = round((sum(abs(model.predict(x_tr)-y_tr))/len(y_tr))[0],3),
                                    mae_val     = round((sum(abs(model.predict(x_vl)-y_vl))/len(y_vl))[0],3),
                                    mae_test    = round((sum(abs(model.predict(x_te)-y_te))/len(y_te))[0],3),
                                    mse_train   = round((sum((model.predict(x_tr)-y_tr)**2)/len(y_tr))[0],3),
                                    mse_val     = round((sum((model.predict(x_vl)-y_vl)**2)/len(y_vl))[0],3),
                                    mse_test    = round((sum((model.predict(x_te)-y_te)**2)/len(y_te))[0],3),
                                    r2_train    = round(r2_score(y_tr,model.predict(x_tr)),3),
                                    r2_val      = round(r2_score(y_vl,model.predict(x_vl)),3),
                                    r2_test     = round(r2_score(y_te,model.predict(x_te)),3),
                                    name_method = name_method,
                                    to_txt      = True,
)

 