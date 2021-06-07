"""
File name: lstm.py
Author: Luigi Cesarini
E-mail: luigi.cesarini@iusspavia.it
Date created: 21 November 2020 
Date last modified: 07 June 2021

####################################################################
PURPOSE:
Perform the task of predicting milk prodiction using a recurrent neural network, 
namely a long short term memory model (LSTM) implemented through tensorflow

"""
# IMPORT MODULE
import os 
import time
import datetime
from datetime import datetime,date
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from itertools import combinations

#TF imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from tensorflow.python.keras.backend import variable

# Set random seed 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Store dates and times for printing purposes
d = datetime.now()   
date = datetime.today()
name_method = 'LSTM'

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

"""
Define a function that scales the partitions of the data and return the scaled partitions
"""
def scale_feature(df_train, df_validation, df_testing, prefix_feat, type_scaling):
    if type_scaling == 'std':
        #Select columns based on feature
        df = df_train.loc[:,df_train.columns.str.contains(prefix_feat)]
        feature = np.array(df)
        mean_f  = feature.mean()
        std_f   = feature.std()
        scaled_training   = (df_train.loc[:,df_train.columns.str.contains(prefix_feat)] - mean_f)/std_f
        scaled_validation = (df_validation.loc[:,df_validation.columns.str.contains(prefix_feat)] - mean_f)/std_f
        scaled_testing    = (df_testing.loc[:,df_testing.columns.str.contains(prefix_feat)] - mean_f)/std_f

        return [scaled_training,scaled_validation,scaled_testing]
    else:
        #Select columns based on feature
        df = df_train.loc[:,df_train.columns.str.contains(prefix_feat)]
        feature = np.array(df)
        min_f  = feature.min()
        max_f   = feature.max()
        scaled_training   = (df_train.loc[:,df_train.columns.str.contains(prefix_feat)] - min_f)/(max_f-min_f)
        scaled_validation = (df_validation.loc[:,df_validation.columns.str.contains(prefix_feat)] - min_f)/(max_f-min_f)
        scaled_testing    = (df_testing.loc[:,df_testing.columns.str.contains(prefix_feat)] - min_f)/(max_f-min_f)

        return [scaled_training,scaled_validation,scaled_testing]
        
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# List of countries
countries = ['Germany','France','Italy','DOM']

# How to scale the data. using 'std' will standardized the data  
type_scaling = 'minmax'

for country in countries:
    # Different lenght of the moving window explored
    # The dominican republic having a shorter record does not allow to test 
    # length of the window larger than 12 
    if country != 'DOM':
        length_window = np.arange(2,24,1) 
    else:
        length_window = np.arange(2,12,1) 

    for window in length_window:
        #Climate variables used as input
        # - spi: Standard precipitation index 
        # - anomaly_thi: Anomaly of the temperature-humdity index
        # - ndvi: Vegetation index

        training_variables  =  ["ndv",
                                "spi",
                                "anomaly_thi",                           
                                ] 
        # Iterate over each combination of input data always using the lagged milk as input
        for n_var in range(len(training_variables)+1): 
            combo = []
            for i,names in enumerate(list(combinations(training_variables,n_var))):
                x = list(names)
                x.append('lagged_milk')
                combo.append(x)
            for var in combo:
                # Features of the model in case. Useful for printing
                type_model = f"LSTM {type_scaling} {window} Window:{[x for x in var]}"
                # Load the whole dataset
                data = pd.read_csv(f"MachineLearning/Milk/data/{country}/input_{country}.txt", sep = "\t", header = 0, index_col = "date")
                # Using the milk of the 1 month lagged milk as input for the model
                data["lagged_milk"] = data.milk.shift(0)
                # Select the months used to train the model. They have to match the period for which the climate data are available
                data = data.loc["1983-01-01":"2019-11-01"]
                # Store the dates in a variable 
                dates = pd.DataFrame(data.index, columns = ["date"])
                # Reordering of the dataframe
                cols = data.columns.tolist()
                cols.remove('milk')
                cols.append('milk')
                data = data[cols]

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
                x_train,y_train = training[training_variables], training[["milk"]]
                x_val,y_val     = validation[training_variables],   validation[["milk"]]
                x_test, y_test  = testing[training_variables],  testing[["milk"]]


                # SEPARATE INPUT VARIABLE BY FEATURES AND SCALE
                x_training   = []
                x_validation = []
                x_testing    = []
                for i in training_variables:
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
                # LSTM Model        
                inputs      = Input(shape = (x_tr.shape[1],x_tr.shape[2]) )
                x           = Bidirectional(LSTM(64, activation='relu', return_sequences=False))(inputs)
                x           = Dropout(0.3)(x)
                x           = Dense(120, activation = "relu")(x)
                outputs     = Dense(1)(x)
                model = keras.Model(inputs=inputs, outputs = outputs)
                # Check model summary in this case when the length of the window is equal to 4
                if window == 4:
                    print(model.summary())

                # Save an image of the network architecture to file    
                tf.keras.utils.plot_model(model,
                                        show_shapes = True,
                                        show_layer_names = True,
                                        to_file="MachineLearning/Milk/output/model_lstm_{}.png".format(int(time.time())))
                """
                Compile the model:
                Mean absolute error used as loss function during the training.
                RMSprop use as optimizer to update the weight of the neural network
                """
                model.compile(
                    loss      = "mae",
                    optimizer = keras.optimizers.RMSprop(),
                    metrics   = ["mae","mse"],
                )

                # Initialize two callbacks
                # The early stopping is used to stop the training once the minimization of the validation loss reaches a plateau, thus, avoiding overfitting.
                # The ReduceLROnPlateau is used to reduce the learning rate of the optimizer when the minimization of the validation loss reaches a plateau
                # The patience represents the number of epochs with no improvement after which training will be stopped or learning rate will be reduced.  
                patience = 2
                callback_reduce_lr = ReduceLROnPlateau(monitor='loss',
                                                        factor=0.1,
                                                        min_lr=1e-8,
                                                        patience=patience,
                                                        verbose=0)

                callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                        patience=patience, verbose=1)

                callbacks = [callback_early_stopping,callback_reduce_lr]

                """ Fitting of the model """
                history = model.fit(x_tr, y_tr,
                            epochs=500,
                            batch_size = 32, 
                            callbacks = callbacks,
                            verbose = 0,
                            shuffle = False,
                            validation_data = (x_vl,y_vl)
                            )

                def plot_time_series():
                    """
                        Plot of the ground truth vs the predicted
                        Step 1: Create df with columns = [ Date, Obs_Milk, Pred_Milk] for both training and testing
                    """
                    # Create the correct data frame of ground truth
                    ground_truth = pd.concat([milk_tr[window:],milk_vl[window:],milk_te[window:]])
                    # Create the correct data frame of predictions
                    predictions  = pd.DataFrame(np.concatenate([model.predict(x_tr),
                                                                model.predict(x_vl),
                                                                model.predict(x_te)]))
                    predictions.index = ground_truth.index
                    # Join the 2 df to obtain 2 columns [Obs Milk, Pred Milk]
                    df = ground_truth.join(predictions)
                    # Move the date in the index into a column
                    df.reset_index(inplace=True)
                    # Rename the columns
                    df = df.rename(columns = {0:"Pred. Milk", 
                                                "milk":"Obs. Milk",
                                                "date":"Date"})
                    # Sort dataframe by dates
                    df = df.sort_values(by =["Date"])

                    # Start the plotting
                    plt.plot(df.Date,df["Obs. Milk"],'-bo', label = "Ground Truth")
                    plt.plot(df.Date,df["Pred. Milk"],'-ro', label = "Predicted train")
                    plt.axvline(x = "{}".format(training.index[window]))
                    plt.axvline(x = "{}".format(training.index[-1]))
                    plt.axvline(x = "{}".format(validation.index[window]), color = "green")
                    plt.axvline(x = "{}".format(validation.index[-1]), color = "green")
                    plt.axvline(x = "{}".format(testing.index[window]), color = "red")
                    plt.axvline(x = "{}".format(testing.index[-1]), color = "red")
                    plt.axvspan(xmin = "{}".format(training.index[window]),
                                xmax = "{}".format(training.index[-1]),
                                facecolor='#2ca02c', alpha=0.05, label = "Training")
                    plt.axvspan(xmin = "{}".format(validation.index[window]),
                                xmax = "{}".format(validation.index[-1]),
                                facecolor='#228B22', alpha=0.25, label = "Validation")
                    plt.axvspan(xmin = "{}".format(testing.index[window]),
                                xmax = "{}".format(testing.index[-1]),
                                facecolor='#FFFF99', alpha=0.25, label = "Testing")
                    plt.xticks(np.arange(0,(x_tr.shape[0]+x_vl.shape[0]+x_te.shape[0]),7),rotation = 90)
                    plt.title("Training and testing with a {} months window for {}".format(window, country))
                    plt.legend()
                    plt.show()
                    # TURN COMMENT OFF TO SAVE THE PLOT TO DISK
                    #plt.savefig("plot/timeseries/{}_{}_{}_{}var_{}".format(country,window,type_model,x_train.shape[1],int(time.time())), dpi = 300)

                plot_time_series()


                print_performances(type_model ,
                                    mae_train = round((sum(abs(model.predict(x_tr)-y_tr))/len(y_tr))[0],3),
                                    mae_val   = round((sum(abs(model.predict(x_vl)-y_vl))/len(y_vl))[0],3),
                                    mae_test  = round((sum(abs(model.predict(x_te)-y_te))/len(y_te))[0],3),
                                    mse_train = round((sum((model.predict(x_tr)-y_tr)**2)/len(y_tr))[0],3),
                                    mse_val   = round((sum((model.predict(x_vl)-y_vl)**2)/len(y_vl))[0],3),
                                    mse_test  = round((sum((model.predict(x_te)-y_te)**2)/len(y_te))[0],3),
                                    r2_train  = round(r2_score(y_tr,model.predict(x_tr)),3),
                                    r2_val    = round(r2_score(y_vl,model.predict(x_vl)),3),
                                    r2_test   = round(r2_score(y_te,model.predict(x_te)),3))