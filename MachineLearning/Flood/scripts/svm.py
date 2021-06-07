
"""
File name: svm.py
Author: Luigi Cesarini
E-mail: luigi.cesarini@iusspavia.it
Date created: 08 July 2020 
Date last modified: 07 June 2021

####################################################################
PURPOSE:
The script train and run the support vector machine models for the flood case.
It also explores the different set of parameters to establish the domain 
of configurations.
The evaluation metrics are printed to disk in csv format.
"""
# MODULE IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from statistics import mean

from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix

from itertools import combinations
from imblearn.over_sampling import RandomOverSampler, SMOTE    

import time
import datetime
################################
#DEFINITION OF CUSTOM FUNCTION
################################
def find(name, path):
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def plot_cm(labels, predictions, p):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", )
    plt.title('Confusion matrix @{:.2f}'.format(p) + ' ' + format(name_DS))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
################################

#Set working directory to the parent of the script folder
#os.chdir('..')
print(os.getcwd())
# STORE THE ACRONYM OF THE METHOD FOR PRINTING REASONS
name_method = "svm"

# Load the whole dataset 
data_dom = pd.read_csv(find('dataset_DOM.txt',os.getcwd()), sep = "\t")
# Shift soil moisture to get the values of the previous day 
data_dom.iloc[:-1,1:5] = data_dom.iloc[:,1:5].shift(-1).iloc[:-1,:]


# Routines that creates all the configurations we want to explore:
# Remember, in this istance, the soil moisture dataset are added 
# directly to the 6 rainfall datasets (e.g., No combination of 2 rainfall
# datasets + 1 soil moisture was used and so on)

num_ds = np.arange(2,7,1)   
name_columns = data_dom.columns.to_list()[5:11]  #Indexing used to create the combination up to six datasets only amid rainfall datasets
list_comb = [0] * len(num_ds)
counter = 0
for k in num_ds:
  counter += 1
  list_comb[(counter - 1)] = list(combinations(name_columns, k))
### Add combination with all RF DS plus SM
list_comb.append([tuple(data_dom.columns.to_numpy()[[1,5,6,7,8,9,10]])])
list_comb.append([tuple(data_dom.columns.to_numpy()[[1,2,5,6,7,8,9,10]])])
list_comb.append([tuple(data_dom.columns.to_numpy()[[1,2,3,5,6,7,8,9,10]])])
list_comb.append([tuple(data_dom.columns.to_numpy()[[1,2,3,4,5,6,7,8,9,10]])])

#The validation of the model was carired out through a stratified k-fold with k=5
# Split of input (i.e.,X) and target (i.e.,y)
X = data_dom.drop(["Output"], axis = 1)
y = data_dom.Output
# Creation of the 5 splits
kf = StratifiedKFold(n_splits=5, random_state = 42, shuffle = True)
kf.get_n_splits(X,y) 
# HERE BEGINS THE TRAINING LOOP
"""
Initialization of the parameters the training iterate over:

Sampling technique to fight class imbalance (sampling_technique): 
1) Unweighted (unw)
2) Class Weight (cw)
3) Oversampling (over)
4) Synthetic Minority over sampling technique (smote)
Kernel function (kernel):
1) Linear (linear)
2) Radial basis function (rbf)
3) Polynomial (poly)
C-Regularization parameter (c_param):
{0.1,1,10,100,500}

The loop over the abovementioned parameters is done for every fold
"""
sampling_technique = ["unw","cw","over","smote"]
kernel = ['linear','rbf','poly']
c_param = [0.1,1,10,100,500]
for C_par in c_param:
    #print(C_par)
    for st in sampling_technique:
        # print(st)
        n_fold = 0 
        for train_index,test_index in kf.split(X,y):
            n_fold += 1 
            #For the i-th fold split the data into training and testing set       
            x_train, x_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            print(f"Event % train: {np.round(np.count_nonzero(y_train)/y_train.shape[0],3)}%")
            print(f"Event % test: {np.round(np.count_nonzero(y_test)/y_test.shape[0],3)}% on {y_test.shape[0]} samples")
            # Scale the data between 0 and 1
            scaler = MinMaxScaler(feature_range=(0,1))
            scaler.fit(x_train.drop(["Date"],axis = 1))
            x_train.iloc[:,1:] = scaler.transform(x_train.drop(["Date"],axis = 1))
            x_test.iloc[:,1:]  = scaler.transform(x_test.drop(["Date"],axis = 1))


            #write csv with dates and observed historical events
            ts_fold = datetime.datetime.today().strftime('%d_%b_%y')   #timestamp beginning fold i-th
            if os.path.isdir(f'MachineLearning/Flood/output/{name_method}'):
                x_test.Date.to_frame().join(y_test).to_csv(f"MachineLearning/Flood/output/{name_method}/dates_training_{C_par}_{st}_{n_fold}_{ts_fold}.csv",
                                                           quoting=0, index=False, index_label=False, header=True, sep='\t')
            else:
                os.mkdir(f'MachineLearning/Flood/output/{name_method}')
                x_test.Date.to_frame().join(y_test).to_csv(f"MachineLearning/Flood/output/{name_method}/dates_training_{C_par}_{st}_{n_fold}_{ts_fold}.csv",
                                            quoting=0, index=False, index_label=False, header=True, sep='\t')




            #### Computes the class weights
            event = 0.5 * len(y_train) / y_train.value_counts()[1]
            no_event = 0.5 * len(y_train) / (len(y_train)-y_train.value_counts()[1])
            # Accordingly to the sampling technique use the appropriate method to treat the data
            # For class weight set the class_weight argument to != None
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
            # ITERATE OVER THE KERNEL TYPE
            for k_i,name_k in enumerate(kernel):
                # print(name_k)
                #### Create classifier
                svcclassifier = SVC(kernel= kernel[k_i],
                                    class_weight=class_weight,
                                    probability = True,
                                    C=C_par,
                                    max_iter= 10e6)
                # Iterate over the combinations with different lenght (i.e., index 0 for 1 dataset combination,
                # index 1 for 2 dataset combinations, etc.)
                for j in np.arange(len(list_comb)):   
                    #Iterate over each combination of datasets of a given length 
                    for z in np.arange(len(list_comb[j])):
                        # Get the names of the datasets comprising the z-th combination of the j-th list of combinations
                        name_DS = [''.join(i) for i in list_comb[j][z]]
                        # Training model
                        fitting = svcclassifier.fit(x_tr[name_DS], y_tr.values.ravel())
                        # Predictions
                        y_pred = svcclassifier.predict_proba(x_test[name_DS])
                        # Initialize the vector of probabilities threshold explored from 0.01 to 0.99 with step 0.01
                        prob_v = np.arange(start=0.01, stop=1, step=0.01)
                        # Initialize lists that will host the values of precision,recall and F1 score
                        pr = [0] * len(prob_v)
                        re = [0] * len(prob_v)
                        f1 = [0] * len(prob_v)
                        # Iterate over the entire range of probabilities threshold to compute the evaluation metrics
                        for k in np.arange(len(prob_v)):
                            tn, fp, fn, tp = confusion_matrix(y_test, y_pred[:, 1] > prob_v[k]).ravel()
                            if tp+fp == 0:
                                pr=0
                            else:
                                pr = tp/(tp+fp)
                            if tp+fn == 0:
                                re=0
                            else:
                                re = tp/(tp+fn)
                            if tp + fn == 0:
                                sp = 0
                            else:
                                sp = tn / (tn + fp)
                            if pr+re == 0:
                                f1[k]=0
                            else:
                                f1[k] = 2 * (pr * re) / (pr + re)
                            # Dictionaries contaninig the information necessary to evaluate the model:
                            # - Architecture of the SVM model,
                            # - N° of the fold,
                            # - Value of the probability threshold   
                            # - Evaluation metrics   
                            # The dictionary created at each iteration will form a row in the csv with the evaluation metric
                            metrics = {
                                'Configuration': f"{C_par}-{st}-{''.join(name_DS[op][2] for op in np.arange(len(name_DS)))}-{kernel[k_i]}",
                                'Fold' : n_fold,
                                'Probabilities': " Pr > " + format(prob_v[k].round(2)),
                                'N° Dataset': format(len(name_DS)),
                                'Precision': [pr],
                                'Sensitivity': [re],
                                'Specificity': [sp],
                                'F1 Score': [f1[k]],
                                'Method': 'SVM',
                                'TP': tp,
                                'TN': tn,
                                'FP': fp,
                                'FN': fn
                            }

                            #Create the dataframe containing the dictionary previously created
                            df_metrics = pd.DataFrame(metrics, columns=['Configuration', 'Probabilities', 'Fold','N° Dataset', 'Precision',
                                                                        'Sensitivity', 'Specificity', 'F1 Score', 'Method', 'TP',
                                                                        'TN', 'FP', 'FN'])
                            # Save the metrics to a csv file
                            df_metrics.to_csv(f"MachineLearning/Flood/output/{name_method}/metrics_{name_method}_{ts_fold}.csv",
                                              quoting=0, index=False, index_label=False, mode='a',
                                              header=False, sep='\t')

                        # Create DataFrame with the prediction provided by the model with the highest F1 score for a given threshold probability
                        # and print it to file
                        pd.DataFrame((y_pred[:, 1] > prob_v[f1 == max(f1)][0]).astype('int'),
                        columns = [f"{C_par}-{st}-{''.join(name_DS[op][2] for op in np.arange(len(name_DS)))}-{n_fold}-{kernel[k_i]}"]).transpose().to_csv(f"MachineLearning/Flood/output/{name_method}/predictions_{name_method}_{ts_fold}.csv",
                        quoting=0, index=True,mode='a', header=False, sep='\t') 



