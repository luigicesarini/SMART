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
from custom_functions import plot_time_series,scale_feature,create_dataset, print_performances, create_dataset_multistep
import warnings
warnings.filterwarnings('ignore')
# Set random seed 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

patience = 4
list_country = [
    "France",
    "Italy",
    "Germany",
    'DOM'
]

country = "DOM"
cc = country[0:3]
forecast_horizon = 12
for country_model in list_country:
    print(country_model)

    if country_model == 'France':
        length_window = [13]
    elif country_model == 'Germany':
        length_window = [13]
    elif country_model == 'Italy':
        length_window = [13]
    else:
        length_window = [13]


    data = pd.read_csv("csv/{}/input_{}.txt".format(country,country), sep = "\t", header = 0, index_col= 'date')
    data["shifted_milk"] = data.milk.shift(0)
    data = data.loc["2009-01-01":"2015-12-01"] # !! REMEMBER: Dropping NA we loose the last month
    dates = pd.DataFrame(data.index, columns = ["date"])
    cols = data.columns.tolist()
    cols.remove('milk')
    cols.append('milk')
    data = data[cols]

    #var = ['shifted_milk']
    for window in length_window:
        training_variables  =  [
                                #"thi",
                                #"ndv"
                                ]   
        for n_var in  np.arange(0,len(training_variables)+1):#range(len(training_variables)):
            combo = []    
            for i,names in enumerate(list(combinations(training_variables,n_var))):
                x = list(names)
                x.append('shifted_milk')
                combo.append(x)
            #var = combo[0]
            for var in combo:
                # SPLITTING
                start_validation = dates.date[dates.date == "2013-01-01"].values[0] #dates.date[int(0.7*len(data))]
                start_testing    = dates.date[dates.date == "2015-01-01"].values[0] #dates.date[int(0.9*len(data))]
                finish_testing   = dates.date[-1:].values[0]


                index_validation  = dates.date[dates.date[dates.date == "2013-01-01"].index-window].values[0]
                index_testing     = dates.date[dates.date[dates.date == "2015-01-01"].index-window].values[0]
                #Splitting
                training   = data.iloc[np.array(dates.date < start_validation)]
                validation = data.iloc[np.array((dates.date >= index_validation)&(dates.date < start_testing))]
                testing    = data.iloc[np.array(dates.date >= index_testing)]

                x_train,y_train = training[var], training[["milk"]]
                x_val,y_val     = validation[var],   validation[["milk"]]
                x_test, y_test  = testing[var],  testing[["milk"]]


                mean_milk = y_train.milk.mean()
                sd_milk   = y_train.milk.std()

                # SEPARATE INPUT VARIABLE BY FEATURES AND SCALE
                # Aninmals and shifted production
                x_training   = []
                x_validation = []
                x_testing    = []
                for i in var:
                    x_training.append(scale_feature(x_train,x_val,x_test,i)[0])
                    x_validation.append(scale_feature(x_train,x_val,x_test,i)[1])
                    x_testing.append(scale_feature(x_train,x_val,x_test,i)[2])

                x_training      = pd.concat(x_training,axis=1)
                x_validation    = pd.concat(x_validation,axis=1)
                x_testing       = pd.concat(x_testing,axis=1)
                #milk
                milk_tr,milk_vl,milk_te = scale_feature(y_train,y_val,y_test,'milk')

                # Splitting into sequences
                # CREATING BATCHES 
                # Training
                x_tr,y_tr   = create_dataset_multistep(x_training,milk_tr,window, forecast_horizon)
                #Validation
                x_vl,y_vl   = create_dataset_multistep(x_validation,milk_vl,window,forecast_horizon)
                #Testing
                x_te,y_te   = create_dataset_multistep(x_testing,milk_te,window,forecast_horizon)

                tf.keras.backend.clear_session()
                if country_model == 'DOM':
                # spiipitation Channel
                    inputs       = keras.layers.Input(shape = (x_tr.shape[1],x_tr.shape[2])) 
                    conv1_1      = keras.layers.Conv1D(filters = 128, kernel_size = int(window/2), activation = 'relu', padding = 'same', name = "CNN_1_1")(inputs)
                    flatten_1    = keras.layers.Flatten(name = "Flat_1")(conv1_1)
                    dense_1      = keras.layers.Dense(100, name = "Dense_1")(flatten_1)
                    drop_1       = keras.layers.Dropout(0.2, name = "Drop_2")(dense_1)
                    outputs_spi = keras.layers.Dense(forecast_horizon, name = "Predictions")(drop_1)

                    model = keras.Model(inputs=inputs, outputs = outputs_spi)
                else:
                    model = keras.models.load_model("trained_model/{}/M4_MS_EVs_{}_W{}_FH12".format(country_model,country_model,window))


                model.compile(
                    loss      = "mse",
                    optimizer = keras.optimizers.RMSprop(lr = 1e-4),
                    metrics   = ["mae","mse"],
                )

                callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                        factor=0.5,
                                                        min_lr=1e-8,
                                                        patience=2,
                                                        verbose=0)

                callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                        patience=patience, verbose=1)

                callbacks = [callback_early_stopping,
                #callback_reduce_lr
                            ]
                if country_model != 'DOM':
                    history = model.fit(x_tr, y_tr,
                                epochs=500,
                                #validation_split = 0.5,
                                batch_size = 32, #int(0.1*x_tr.shape[0]),
                                callbacks = callbacks,
                                verbose = 0,
                                shuffle = False,
                                validation_data = (x_vl,y_vl)
                                )
                    
                # Year 2020 up to august, therefore we take the 13 months prior to august 2019
                # to predict 08-2019/08-2020

                input  = np.array(x_testing.loc["2013-12-01":"2014-12-01", x_testing.columns.str.contains("|".join([x for x in var]))]).reshape(1,window,x_testing.shape[1])
                output = (np.array(data.loc["2015-01-01":"2015-12-01", "milk"]).reshape(1,forecast_horizon,1)-mean_milk)/sd_milk
                output_2 = np.array(milk_te.loc["2015-01-01":"2015-12-01"]).reshape(1,forecast_horizon,milk_te.shape[1])

                yearly_truth = np.sum(output*sd_milk+mean_milk)
                print("Yearly truth:{}".format(yearly_truth))
                yearly_pred  = sum(model.predict(input).reshape(forecast_horizon,1)*sd_milk+mean_milk)
                print("Yearly error in millions of liter {}\nError as a percentage of yearly milk production {}%".format(round(abs(yearly_truth-yearly_pred).item(),2),
                                                                                                                                    abs(round(((yearly_truth-yearly_pred)/yearly_truth).item()*100,2))))

                if country_model == 'DOM':
                    plt.plot(model.predict(input).transpose()*sd_milk+mean_milk, '-ko', label = f'DOM Model')
                elif country_model == 'Germany':
                    print(f'Train {country_model}')
                    plt.plot(model.predict(input).transpose()*sd_milk+mean_milk, '-go', label = f'Train {country_model}')
                elif country_model == 'France':
                    plt.plot(model.predict(input).transpose()*sd_milk+mean_milk, '-mo', label = f'Train {country_model}')
                elif country_model == 'Italy':
                    plt.plot(model.predict(input).transpose()*sd_milk+mean_milk, '-ro', label = f'Train {country_model}')

                    plt.plot(output.reshape(forecast_horizon,1)*sd_milk+mean_milk, '-bo', label = 'Observed')
                    #plt.plot(output_2.reshape(forecast_horizon,1)*sd_milk+mean_milk, '--ro', label = 'Truth 2')
                    plt.title(f"Forecast in the Dominican Republic over period Jan 2015 - Dec 2015\n Input Variables: THI and lag-1 month milk")
                    plt.ylabel("Milk in thousands of tonnes")
                    plt.xticks(np.arange(0,forecast_horizon,1),
                            labels = pd.date_range(start = '01/01/2015', end = '12/01/2020', freq='MS').strftime("%b-%Y")[np.arange(0,forecast_horizon,1)],
                            rotation = 0)
                    plt.grid(alpha = 0.5)



                monthly_mse = []
                monthly_mae = []
                monthly_r2  = []
                for i in range(forecast_horizon):
                    mae_forecast = round(mean_absolute_error(output[:,i,0],model.predict(input)[:,i]),3)
                    mse_forecast = round(mean_squared_error(output[:,i,0],model.predict(input)[:,i]),3)
                    r2_forecast  = round(r2_score(output[:,i,0],model.predict(input)[:,i]),3)
                    monthly_mse.append(mse_forecast)
                    monthly_mae.append(mae_forecast)
                    monthly_r2.append(r2_forecast)
                
                #print(output.shape)
                #print(model.predict(input).shape)
                print(country_model)
                print(monthly_mse)
                print(monthly_mae)
                # print(f'{country} MSE: {np.array(monthly_mse).mean()}')
                # print(f'{country} MAE: {np.array(monthly_mae).mean()}')
                #print(monthly_r2)

plt.legend()
plt.show()

