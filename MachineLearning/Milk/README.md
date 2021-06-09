## Forecast of milk production

The folder "Milk" contains the scripts of the developed model along with the data necessary to replicate the training of the model. The folder "output" includes also some examples of the output generated.

Detailed information about the methodology, the data and the assumptions used in developing the models can be found in the final report of the project at:

[https://www.gfdrr.org/en/challengefund](https://www.gfdrr.org/en/challengefund)  

The folder "scripts" contains the type of model generated for the predicion of milk production:

- **arima.py** : contains the ARIMA model used as a baseline for the ML models
- **lstm.py** : contains the code used to develop the Long Short Term Memory models   
-  **cnn\_ssp.py** : contains the code used to develop the one-dimensional convolutional neural network for single-step prediction for european countries.
-  **cnn\_ssp\_transfer.py** : contains the code used to train the model already used in Europe for single-step prediction in the Dominican Republic (*transfer learning*)
-  **cnn\_msp.py** : contains the code used to develop the one-dimensional convolutional neural network for multi-step prediction for european countries and the Dominican Republic.
-  **cnn\_msp\_transfer.py** : contains the code used to train the model already used in Europe for multi-step prediction in the Dominican Republic (*transfer learning*)