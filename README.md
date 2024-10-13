# Electricity Load Forecasting Project

This project predicts electricity load using various machine learning and deep learning models, including LSTM, CNN, MLP, Stacked LSTM, XGBoost, and Encoder-Decoder models. It uses weather data as input to forecast future electricity demands.

## Project Structure

```
load-forecasting/
│
├── data/
│   └── clean_data.csv            
│
├── src/
│   ├── data_preprocessing.py     
│   ├── feature_engineering.py    
│   ├── correlation_analysis.py   
│   ├── multivariate_data.py      
│   └── model_training.py        
│
├── models/
│   ├── lstm_model.py             
│   ├── stacked_lstm_model.py     
│   ├── cnn_model.py             
│   ├── mlp_model.py              
│   ├── xgboost_model.py          
│   └── encoder_decoder_model.py 
│
├── config.yaml                  
├── requirements.txt              
├── README.md                   
└── main.py                      
```



## Requirements

To install the required libraries, use the following command:

```bash
pip install -r requirements.txt
```

1. **Configure the hyperparameters**:
Modify the `config.yaml` file to tune hyperparameters such as learning rates, number of layers, batch sizes, etc.

3. *Run the project*:
To execute the models and train them, simply run the following command:


```
python main.py
```

This will preprocess the data, build models, train them, and save the best-performing models.

4. Checkpoints:
Each deep learning model will automatically save the best checkpoint in .h5 format.


### Available Models:
The following models are implemented in the project:

`LSTM`: For time series forecasting using sequential data.

`Stacked LSTM`: A deeper version of the LSTM model.

`CNN`: For learning spatial features from time series data.

`MLP`: A feedforward neural network with multiple hidden layers.

`XGBoost`: A gradient boosting model for regression tasks.

`Encoder-Decoder LSTM`: For sequence-to-sequence learning.



### Future Enhancements
Add additional model architectures such as GRU or Transformer models.
Implement hyperparameter tuning using tools like Optuna or GridSearchCV.
Provide visualizations of model performance for better interpretability.

---
