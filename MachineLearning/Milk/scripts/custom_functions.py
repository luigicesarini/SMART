# PLOT TIMESERIES
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import datetime

def plot_time_series(window,inputs_training,inputs_validation,inputs_testing,
                     milk_tr,milk_vl,milk_te,
                     training,validation,testing,
                     model,country):
    """
        Plot of the ground truth vs the predicted
        Step 1: Create df with columns = [ Date, Obs_Milk, Pred_Milk] for both training and testing
    """
    # Create the correct data frame of ground truth
    ground_truth = pd.concat([milk_tr[window:],milk_vl[window:],milk_te[window:]])
    # Create the correct data frame of predictions
    predictions  = pd.DataFrame(np.concatenate([model.predict(inputs_training),
                                                model.predict(inputs_validation),
                                                model.predict(inputs_testing)]))
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
    plt.xticks(np.arange(0,(training.shape[0]+validation.shape[0]+testing.shape[0]),7),rotation = 90)
    plt.title("Training and testing with a {} months window for {}".format(window, country))
    plt.legend()
    plt.show()

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

def create_dataset_multistep(X, y, time_steps=1, forecast_horizon = 3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - forecast_horizon):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[(i + time_steps):(i + time_steps + forecast_horizon)].values)
    return np.array(Xs), np.array(ys)

"""
Define a function that prints the performances of the models either to disk or to the console

"""

def print_performances(type_model,country,id_model,
                       mse_train, mse_val, mse_test,
                       mae_train, mae_val, mae_test,
                       r2_train, r2_val, r2_test,name_method,
                       to_txt = True
                       ):
    import datetime
    import os
    prepare_print = {
        "Model"     :   type_model,
        'country'   :   country,
        "date"      :   datetime.datetime.now().ctime(),
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
        pd.DataFrame(prepare_print, index = [0]).to_csv(f"MachineLearning/Milk/output/metrics_milk_production_{name_method}_{datetime.datetime.today().strftime('%d_%m_%Y')}.txt",
                        quoting=0, index=False, index_label=False, mode='a', header=False, sep='\t')
    else:
        print(prepare_print)

def create_trainable_data(country, window,
                          date_start_training,
                          date_start_validation, date_finish_validation,
                          date_start_testing,
                          input_variables,
                          partition):
    """
    This functions as purpose of preparing the data to be suitable to fed the model.
    It first laods the data for a given country, creates the column of past milk and
    split the data into training, validation and testing set based on the provided starting
    dates of validation and testing.
    # Arguments:
        - country: the country we want to prepare the data for.
        - window: length of the moving window (i.e. the length of each batch)
        - date_start_validation: when do we want the validation period to start. 
          Based on this value, also the training set is created.
        - date_start_testing: the date from which the test set starts. This also indicates 
          the end of the validation partition.
        - input_variable: the varaibles we are using from the poll we have. (i.e. thi,ndvi,spi,past_milk)
        - partition: indicates the set we are creating
    """
    df_input_data  = pd.read_csv("csv/{}/input_{}.txt".format(country,country), sep = "\t", header = 0, index_col= 'date')
    df_input_data["shifted_milk"] = df_input_data.milk.shift(0)
    df_input_data = df_input_data.loc["1982-01-01":"2019-12-01"] # !! REMEMBER: Dropping NA we loose the last month
    dates = pd.DataFrame(df_input_data.index, columns = ["date"])
    cols = df_input_data.columns.tolist()
    cols.remove('milk')
    cols.append('milk')
    df_input_data = df_input_data[cols]

    # SPLITTING
    start_validation  = dates.date[dates.date == date_start_validation].values[0] #dates.date[int(0.7*len(data))]
    finish_validation = dates.date[dates.date == date_finish_validation].values[0] #dates.date[int(0.7*len(data))]
    start_testing     = dates.date[dates.date == date_start_testing].values[0] #dates.date[int(0.9*len(data))]
    #finish_testing   = dates.date[-1:].values[0]
    index_training    = dates.date[dates.date[dates.date == date_start_training].index-window].values[0]
    index_validation  = dates.date[dates.date[dates.date == date_start_validation].index-window].values[0]
    index_testing     = dates.date[dates.date[dates.date == date_start_testing].index-window].values[0]
    #Splitting
    training   = df_input_data.iloc[np.array((dates.date >= index_training)&(dates.date <= start_validation))]
    validation = df_input_data.iloc[np.array((dates.date >= index_validation)&(dates.date <= finish_validation))]
    testing    = df_input_data.iloc[np.array(dates.date >= index_testing)]

    if partition == "training":
        X_,y_ = training[input_variables], training[['milk']]
    elif partition == "validation":
        X_,y_ = validation[input_variables], validation[['milk']]
    else:
        X_,y_ = testing[input_variables], testing[['milk']]

    return X_,y_
