"""
Here we perform the task of predicting milk prodiction 
with some models that are going to be used as benchamrk for
the deep learning model:
0) DUMB: milk production of the next month is equal to the production of the previous month
1) ARIMA: past milk production
2) ARIMAX:past milk prodution & animals
3) LSTM: past milk prodution & animals
4) CNN1D: past milk prodution & animals
"""
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
forecast_horizon = 12
length_window = np.arange(13,14,1)

# 4b) CNN 1D 6 variables:
for country in list_country:
    for window in length_window:
        training_variables  =  ["spi",
                                "anomaly_thi",
                                "ndv"
                                ]   
        for n_var in  np.arange(0,1):#range(len(training_variables)):
            combo = []    
            for i,names in enumerate(list(combinations(training_variables,n_var))):
                x = list(names)
                x.append('shifted_milk')
                combo.append(x)
            #var = combo[0]
            for var in combo:
                if country == 'DOM':
                    data = pd.read_csv("csv/{}/input_{}.txt".format(country,country), sep = "\t", header = 0, index_col= 'date')
                    data["shifted_milk"] = data.milk.shift(0)
                    data = data.loc["2009-01-01":"2015-12-01"] # !! REMEMBER: Dropping NA we loose the last month
                    dates = pd.DataFrame(data.index, columns = ["date"])
                    cols = data.columns.tolist()
                    cols.remove('milk')
                    cols.append('milk')
                    data = data[cols]
                else:            
                    data = pd.read_csv("csv/{}/input_{}.txt".format(country,country), sep = "\t", header = 0, index_col= 'date')
                    #data.loc[:,"hr.10026":"temp.9915"] = data.loc[:,"hr.10026":"temp.9915"].shift(-1) 
                    #data.loc[:,"Ita_temp":"Ita_ndvi"] = data.loc[:,"Ita_temp":"Ita_ndvi"].shift(0) 
                    data["shifted_milk"] = data.milk.shift(0)
                    data = data.loc["1982-02-01":"2019-12-01"] # !! REMEMBER: Dropping NA we loose the last month
                    dates = pd.DataFrame(data.index, columns = ["date"])
                    cols = data.columns.tolist()
                    cols.remove('milk')
                    cols.append('milk')
                    data = data[cols]

                type_model = "CNN1 M4 {} W:{} {}".format([x for x in var],window, country )

                # SPLITTING
                start_validation = dates.date[dates.date == "2009-01-01"].values[0] #dates.date[int(0.7*len(data))]
                start_testing    = dates.date[dates.date == "2016-01-01"].values[0] #dates.date[int(0.9*len(data))]
                finish_testing   = dates.date[-1:].values[0]

                index_validation  = dates.date[dates.date[dates.date == "2009-01-01"].index-window].values[0]
                index_testing     = dates.date[dates.date[dates.date == "2016-01-01"].index-window].values[0]
                #Splitting
                training   = data.iloc[np.array(dates.date < start_validation)]
                validation = data.iloc[np.array((dates.date >= index_validation)&(dates.date <= start_testing))]
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
                # Build the model
                """
                Rewrite completely the model creating a multihead model with a cnn for each features
                    - Check which shapes keras.layers.concatenate() is able to take
                """
                tf.keras.backend.clear_session()

                # spiipitation Channel
                inputs       = keras.layers.Input(shape = (x_tr.shape[1],x_tr.shape[2])) 
                conv1_1      = keras.layers.Conv1D(filters = 128, kernel_size = int(window/2), activation = 'relu', padding = 'same', name = "CNN_1_1")(inputs)
                flatten_1    = keras.layers.Flatten(name = "Flat_1")(conv1_1)
                dense_1      = keras.layers.Dense(100, name = "Dense_1")(flatten_1)
                drop_1       = keras.layers.Dropout(0.2, name = "Drop_2")(dense_1)
                outputs_spi = keras.layers.Dense(forecast_horizon, name = "Predictions")(drop_1)

                model = keras.Model(inputs=inputs, outputs = outputs_spi)

                #model.summary()
                #tf.keras.utils.plot_model(model,show_shapes = True,show_layer_names = True,to_file="model_{}.png".format(int(time.time())))

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

                history = model.fit(x_tr, y_tr,
                            epochs=500,
                            #validation_split = 0.5,
                            batch_size = 32, #int(0.1*x_tr.shape[0]),
                            callbacks = callbacks,
                            verbose = 0,
                            shuffle = False,
                            validation_data = (x_vl,y_vl)
                            )

                model.save('trained_model/{}/M4_MS_EV{}_{}_W{}_FH{}'.format(country,"".join([x[0] for x in var]),country,window,forecast_horizon))
                
                # TRaining  plot
                # plt.subplot(121)
                # plt.plot(history.history['loss'], '-r', label = "Training loss")
                # plt.plot(history.history['val_loss'], '-b', label = "Validation loss")
                # plt.title("Training loss vs Validation loss")
                # plt.legend()
                # plt.subplot(122)
                # plt.plot(history.history['mae'], '-r', label = "Training MAE")
                # plt.plot(history.history['val_mae'], '-b', label = "Validation MAE")
                # plt.title("Training MAE vs Validation MAE")
                # plt.legend()
                # plt.show()


                # print_performances(type_model,country, id_model = "M4_multi_step",
                #                     mae_train = round((sum(abs(model.predict(x_tr)-y_tr))/len(y_tr))[0],3),
                #                     mae_val   = round((sum(abs(model.predict(x_vl)-y_vl))/len(y_vl))[0],3),
                #                     mae_test  = round((sum(abs(model.predict(x_te)-y_te))/len(y_te))[0],3),
                #                     mse_train = round((sum((model.predict(x_tr)-y_tr)**2)/len(y_tr))[0],3),
                #                     mse_val   = round((sum((model.predict(x_vl)-y_vl)**2)/len(y_vl))[0],3),
                #                     mse_test  = round((sum((model.predict(x_te)-y_te)**2)/len(y_te))[0],3),
                #                     r2_train  = round(r2_score(y_tr,model.predict(x_tr)),3),
                #                     r2_val    = round(r2_score(y_vl,model.predict(x_vl)),3),
                #                     r2_test   = round(r2_score(y_te,model.predict(x_te)),3),
                #                     to_txt    = False)

                # if round((sum(abs(model.predict(x_te)-y_te))/len(y_te))[0],3) < 0.14:
                #     plot_time_series(window,x_tr,x_vl,x_te,
                #                                 milk_tr,milk_vl,milk_te,
                #                                 training,validation,testing,model,country)

                # monthly_mse = []
                # monthly_mae = []
                # monthly_r2  = []
                # for i in range(forecast_horizon):
                #     mae_forecast = round(mean_absolute_error(y_te[:,i,0],model.predict(x_te)[:,i]),3)
                #     mse_forecast = round(mean_squared_error(y_te[:,i,0],model.predict(x_te)[:,i]),3)
                #     r2_forecast  = round(r2_score(y_te[:,i,0],model.predict(x_te)[:,i]),3)
                #     monthly_mse.append(mse_forecast)
                #     monthly_mae.append(mae_forecast)
                #     monthly_r2.append(r2_forecast)

                # print(monthly_mse,monthly_mae,monthly_r2)
                # print(sd_milk*np.array(monthly_mae))
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
                
                # # for i in np.random.randint(0,y_te.shape[0],size = 3):
                # #     print(i)
                i = y_te.shape[0]-1
                plt.plot(model.predict(x_te)[i].reshape(forecast_horizon,1)*sd_milk+mean_milk, '-go', label = 'predict')
                #plt.plot(model.predict(one_batch).reshape(forecast_horizon,1), '-bo', label = 'predict')
                plt.plot((y_te[i]*sd_milk)+mean_milk, '-ro', label = 'truth')
                plt.title("{} forecast on: {}/{}".format(country,x_test.index[(window+i)],x_test.index[(window+i+forecast_horizon-1)]))
                plt.ylabel("Milk in thousands of tonnes")
                plt.xticks(np.arange(0,len(y_te[i]),1),
                           labels = pd.date_range(start = '12/01/2016', end = '12/01/2019', freq='MS').strftime("%b-%Y")[np.arange(0,len(y_te[i]),1)],
                           rotation = 0)
                plt.legend()
                plt.show()

                # yearly_truth = sum((y_te[i]*sd_milk)+mean_milk)
                # print("Yearly truth:{}".format(yearly_truth))
                # yearly_pred  = sum(model.predict(x_te)[i].reshape(forecast_horizon,1)*sd_milk+mean_milk)
                # print("Yearly error in millions of liter {}\nError as a percentage of yearly milk production {}%".format(round(abs(yearly_truth-yearly_pred).item(),2),
                #                                                                                                          abs(round(((yearly_truth-yearly_pred)/yearly_truth).item()*100,2))))
