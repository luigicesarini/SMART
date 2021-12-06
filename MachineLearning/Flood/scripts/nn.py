
"""
File name: nn.py
Author: Luigi Cesarini
E-mail: luigi.cesarini@iusspavia.it
Date created: 09 July 2020 
Date last modified: 07 June 2021

####################################################################
PURPOSE:
The script train and run the neural networks models for the flood case.
It also explores the different set of parameters to establish the domain 
of configurations.
The evaluation metrics are printed to disk in csv format.

For detailed information about the methodology, the data and the methods used to build
this models, please check the related article at:
https://nhess.copernicus.org/preprints/nhess-2020-220/ 
"""
# MODULE IMPORTS
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.random.set_seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from itertools import combinations

import os
import time
import datetime
import matplotlib.pyplot as plt

################################
#DEFINITION OF CUSTOM FUNCTION
################################
# This function is used to find the path to a specific file
def find(name, path):
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
# This function is  used to plot the confusion matrix for a given probability threshold
def plot_cm(labels, predictions, p):
  import seaborn as sns
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5, 5))
  sns.heatmap(cm, annot=True, fmt="d", )
  plt.title('Confusion matrix @{:.2f}'.format(p) + ' ' + format(name_DS))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
################################
#Set working directory to the parent directory of the project:"SMART"
#os.chdir("../../../")
print(os.getcwd())
#os.chdir(<SMART>)
# Create an early stopping callback. This callback is used to stop the training once
# the minimization of the validation loss reaches a plateau, thus, avoiding overfitting.
es = EarlyStopping(monitor='val_loss', min_delta=1e-20, verbose=1, patience=100)
# STORE THE ACRONYM OF THE METHOD FOR PRINTING PURPOSES
name_method = "nn"
####### Load the whole datasets
data_dom = pd.read_csv(find('dataset.txt',os.getcwd()), sep = "\t")
# Shift soil moisture to get the values of the previous day 
data_dom.iloc[:-1,1:5] = data_dom.iloc[:,1:5].shift(-1).iloc[:-1,:]
# Split data for training and testing (75/25). A percentage of the training set will later be used 
# to validate the training process
x_train, x_test, y_train, y_test = train_test_split(data_dom.drop(['Output'], axis=1),
                                                    data_dom[['Output']],
                                                    test_size=0.25,
                                                    random_state=42)

                            
# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train.drop(["Date"],axis = 1)	)
x_train.iloc[:,1:] = scaler.transform(x_train.drop(["Date"],axis = 1)	)
x_test.iloc[:,1:]  = scaler.transform(x_test.drop(["Date"],axis = 1)	)

#write csv with dates and observed historical events
timestamp = datetime.datetime.today().strftime('%d_%b_%y')   #timestamp beginning fold i-th
if os.path.isdir(f'MachineLearning/Flood/output/{name_method}'):
    x_test.Date.to_frame().join(y_test).to_csv(f"MachineLearning/Flood/output/{name_method}/dates_training_{timestamp}.csv",
                                                quoting=0, index=False, index_label=False, header=True, sep='\t')
else:
    os.mkdir(f'MachineLearning/Flood/output/{name_method}')
    x_test.Date.to_frame().join(y_test).to_csv(f"MachineLearning/Flood/output/{name_method}/dates_training_{timestamp}.csv",
                                quoting=0, index=False, index_label=False, header=True, sep='\t')

#### Computes the class weights
event = 0.5 * len(y_train) / y_train.Output.value_counts()[1]
no_event = 0.5 * len(y_train) / (len(y_train)-y_train.Output.value_counts()[1])

# Routines that creates all the combinations of dataset we want to explore:
# Remember, in this istance, the soil moisture dataset are added 
# directly to the 6 rainfall datasets (e.g., No combination of 2 rainfall
# datasets + 1 soil moisture was used and so on)
num_ds = np.arange(2,7,1)
name_columns = x_train.columns.to_list()[5:11]
list_comb = [0] * len(num_ds)
counter = 0
for k in num_ds:
  counter += 1
  list_comb[(counter - 1)] = list(combinations(name_columns, k))
### Add combination with all RF DS plus SM
list_comb.append([tuple(x_train.columns.to_numpy()[[1,5,6,7,8,9,10]])])
list_comb.append([tuple(x_train.columns.to_numpy()[[1,2,5,6,7,8,9,10]])])
list_comb.append([tuple(x_train.columns.to_numpy()[[1,2,3,5,6,7,8,9,10]])])
list_comb.append([tuple(x_train.columns.to_numpy()[[1,2,3,4,5,6,7,8,9,10]])])

# HERE BEGINS THE TRAINING LOOP
"""
Initialization of the parameters the training iterate over:

Sampling technique to fight class imbalance (sampling_technique): 
1) Unweighted (unw)
2) Class Weight (cw)
3) Oversampling (over)
4) Synthetic Minority over sampling technique (smote)
Number of hidden layers (num_hidden):
- num_hidden = range(1,9)
Number of hidden nodes (hidden_nodes):
- hidden_nodes = 2nl+1 : 2nl+9 ## nl = num_hidden
Activation function (hidden_act):
1) Relu
2) Tanh

The training for all the configurations is obtained by iterating over the abovementioned parameters
"""
sampling_technique = ["unw","cw","over","smote"]
num_hidden = np.arange(9)   # Number of hidden layer
hidden_nodes = [2**(numero_layer+1) for numero_layer in num_hidden]
hidden_act = ["relu","tanh"] # Activation function
loss = [keras.losses.BinaryCrossentropy()]  # Loss fucntion the algorithm wants to minimize
optimizer = [keras.optimizers.Adam(lr = 0.001)]            #list of optimizers used to update the weight of the NN  
# Number of iteration. The training will not reach the max amount because of the call back previously defined                            
epoch = 4000 
batch_size = 64**2
validation_split = 0.2 #Percentage of the training set used to validate the model during the traininig proces (i.e., 15% of the original dataset)
call_back = [es]

for st in sampling_technique:
  """
  Four type of sampling: pristine data,class weight, over-sampling, smote
  Here we check all the condition realtive to the type of sampling we are using
  """
  print(st)

  if st == "cw":
    class_weight = {0: no_event,
                    1: event}
  else:
    class_weight = None

  if st == "over":
    ov_s = RandomOverSampler(random_state=42)
    x_tr, y_tr = ov_s.fit_resample(x_train.drop(["Date"],axis = 1),y_train)
  elif st == "smote":
    sm = SMOTE(random_state=42)
    x_tr, y_tr = sm.fit_resample(x_train.drop(["Date"],axis = 1),y_train)
  else:
    x_tr = x_train.drop(["Date"],axis = 1)				
    y_tr = y_train 


  """
  Herein we iterate over several dataset combination:
  - All 6 combination of rainfall dataset 
  - Adding one by one the soil moisture layer to the 6 rainfall datasets
    """
  for j in np.arange(len(list_comb)):
    """
    Here we iterate over the number of combination i.e. 2DS,3DS,...,9DS,10DS
    """        
    for z in np.arange(len(list_comb[j])):
      """
      Iterate over each combination of the dataset composition
      """
      name_DS = [''.join(i) for i in list_comb[j][z]]
      
      for J in num_hidden:
        
        for n,a in enumerate(hidden_act):
          
          """
          Model Architecture: investigate different size of neural network 
          (#layers and # of nodes) and different act function
          The model is written using Tensorflow's Functional API. Additional information can be found at:
          https://www.tensorflow.org/guide/keras/functional?hl=en
          """
          # Cleans every model that might have still remained in memory
          tf.keras.backend.clear_session()  
          #Input layer
          inputs = keras.Input(shape=x_tr[name_DS].shape[1])
          #Hidden layers
          # The loop creates the NN architecture with increasing number of hidden layer and 
          # hidden nodes according to the J index 
          for k in np.arange(J+1):
            if k == 0:
              x = layers.Dense(hidden_nodes[J-k],activation=hidden_act[n])(inputs)
              x = layers.Dropout(hidden_nodes[J-k]*1/(k+2)/hidden_nodes[J-k])(x)             
            else:
              x = layers.Dense(hidden_nodes[J-k],activation=hidden_act[n])(x)
              x = layers.Dropout(hidden_nodes[J-k]*1/(k+2)/hidden_nodes[J-k])(x)

          #Output layer
          outputs = layers.Dense(1,activation="sigmoid")(x)

          #write the model
          model = keras.Model(inputs=inputs, outputs=outputs, name="nn_flood")	
          # Print hte summary of the architecture to screen
          model.summary()

          """Model Compiler"""
          model.compile(
            loss= loss[0], 
            optimizer=optimizer[0], 
            metrics=[keras.metrics.Precision()] # We use precision as an evaluate metrics during training
          )

          """Model Fitting"""
          history = model.fit(
            x=x_tr[name_DS],
            y=y_tr,
            batch_size=batch_size,
            epochs = epoch,
            validation_split = 0.2,
            verbose = 0,
            class_weight = class_weight,
            callbacks = es
          )
          ###### MODEL EVALUATION  
          # Make the predictions using the testing set
          predictions = model.predict(x_test[name_DS])

          # Plot Training history to check for overfitting
          plt.plot(history.history['loss'])
          plt.plot(history.history['val_loss'])
          plt.title('model loss')
          plt.ylabel('loss')
          plt.xlabel('epoch')
          plt.legend(['train', 'test'], loc='upper left')
          plt.show()


          # Initialize the vector of probabilities threshold explored from 0.01 to 0.99 with step 0.01
          prob_v = np.arange(start=0.01, stop=1, step=0.01)
          # Initialize lists that will host the values of precision,recall and F1 score
          pr = [0] * len(prob_v)
          re = [0] * len(prob_v)
          f1 = [0] * len(prob_v)
          # Iterate over the entire range of probabilities threshold to compute the evaluation metrics
          for k in np.arange(len(prob_v)):
              tn, fp, fn, tp = confusion_matrix(y_test, predictions > prob_v[k]).ravel()
              print(confusion_matrix(y_test, predictions > prob_v[k]).ravel())
              if tp + fp == 0:
                  pr = 0
              else:
                  pr = tp / (tp + fp)
              if tp + fn == 0:
                  re = 0
              else:
                  re = tp / (tp + fn)
              if tp + fn == 0:
                  sp = 0
              else:
                  sp = tn / (tn + fp)
              if pr + re == 0:
                  f1[k] = 0
              else:
                  f1[k] = 2 * (pr * re) / (pr + re)
              # Dictionaries contaninig the information necessary to evaluate the model:
              # - Architecture of the SVM model,
              # - N° of the fold,
              # - Value of the probability threshold   
              # - Evaluation metrics   
              # The dictionary created at each iteration will form a row in the csv with the evaluation metric
              metrics = {
                  #Configuration provides the name indicating the model's configuration trained. it reports:
                  # - The sampling technique
                  # - The input datasets used
                  # - The number of hidden layers
                  # - The activation function 
                  # - The number of epoch the training used to train the model
                  'Configuration': f"{st}-{''.join(name_DS[op][2] for op in np.arange(len(name_DS)))}-{J}-{hidden_act[n]}-{history.epoch[-1]}",
                  'Probabilities': " Pr > " + format(prob_v[k].round(2)),
                  'N° Dataset': format(len(name_DS)),
                  'Precision': pr,
                  'Sensitivity': re,
                  'Specificity': sp,
                  'F1 Score': f1[k],
                  'Method': 'nn',
                  'TP': tp,
                  'TN': tn,
                  'FP': fp,
                  'FN': fn
              }
              #Create the dataframe containing the dictionary previously created
              df_metrics = pd.DataFrame(metrics, columns=['Configuration', 'Probabilities', 'N° Dataset', 'Precision',
                                                                  'Sensitivity', 'Specificity', 'F1 Score', 'Method', 'TP',
                                                                  'TN', 'FP', 'FN'],
                                                index=[0])
              # Save the metrics to a csv file
              oggi = datetime.datetime.today() #Timestamp for file name
              df_metrics.to_csv(f"MachineLearning/Flood/output/{name_method}/metrics_{''.join(name_DS[op][2] for op in np.arange(len(name_DS)))}_{st}_{oggi.strftime('%d_%b_%y')}.csv", 
              quoting=0, index=False, index_label=False, mode='a', header=False, sep='\t')

          # Create DataFrame with the prediction provided by the model with the highest F1 score for a given threshold probability
          df_predictions = pd.DataFrame((predictions > prob_v[f1 == max(f1)][len(prob_v[f1 == max(f1)]) - 1]).astype('int'),
                                        columns = [f"{st}-{''.join(name_DS[op][2] for op in np.arange(len(name_DS)))}-{J}-{hidden_act[n]}"]).transpose()
          # Print the predictions to csv                    
          df_predictions.to_csv(f"MachineLearning/Flood/output/{name_method}/predictions_{name_method}_{oggi.strftime('%d_%b_%y')}.csv",
                                quoting=0, index=True,mode='a', header=False, sep='\t') 


          ##########################################################
          # PLOT CONFUSION MATRIX FOR A GIVEN PROBABILITY THRESHOLD
          ##########################################################

          plot_cm(y_test,predictions, 0.6)
          

