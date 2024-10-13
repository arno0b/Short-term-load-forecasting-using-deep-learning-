import os
import yaml
from src.data_preprocessing import load_and_preprocess_data, generate_time_features, generate_weekend_feature, split_and_normalize_data
from src.correlation_analysis import calculate_correlation, plot_correlation_matrix
from models.lstm_model import build_lstm_model
from models.stacked_lstm_model import build_stacked_lstm_model
from models.cnn_model import build_cnn_model
from models.mlp_model import build_mlp_model
from models.encoder_decoder_model import build_encoder_decoder_model
from models.xgboost_model import train_xgboost_model
from src.model_training import fit_model_with_checkpoint

# Create a directory for saving models if it doesn't exist
save_dir = "models/checkpoints/"
os.makedirs(save_dir, exist_ok=True)

# Load the configuration file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Extract config values
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
lstm_units = config['lstm']['units']
learning_rate = config['lstm']['learning_rate']
outlier_columns = config['preprocessing']['outlier_columns']

# Load and preprocess the data
df = load_and_preprocess_data(config['data']['file_path'], outlier_columns)
df = generate_time_features(df)
df = generate_weekend_feature(df)

# Correlation analysis
corr_matrix = calculate_correlation(df)
plot_correlation_matrix(corr_matrix)

# Split and normalize data
X_train, X_val, y_train, y_val, scaler_X, scaler_y = split_and_normalize_data(df)

# Train XGBoost model and save in JSON format
xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
save_xgboost_model(xgb_model, os.path.join(save_dir, "xgboost_model"))

input_shape = (24, X_train.shape[1]) 

# Train LSTM model and save in H5 format
lstm_model = build_lstm_model(input_shape)
fit_model_with_checkpoint(lstm_model, (X_train, y_train), (X_val, y_val), os.path.join(save_dir, "lstm_model"))

# Train Stacked LSTM model and save in H5 format
stacked_lstm_model = build_stacked_lstm_model(input_shape)
fit_model_with_checkpoint(stacked_lstm_model, (X_train, y_train), (X_val, y_val), os.path.join(save_dir, "stacked_lstm_model"))

# Train CNN model and save in H5 format
cnn_model = build_cnn_model(input_shape)
fit_model_with_checkpoint(cnn_model, (X_train, y_train), (X_val, y_val), os.path.join(save_dir, "cnn_model"))

# Train MLP model and save in H5 format
mlp_model = build_mlp_model(input_shape)
fit_model_with_checkpoint(mlp_model, (X_train, y_train), (X_val, y_val), os.path.join(save_dir, "mlp_model"))

# Train Encoder-Decoder LSTM model and save in H5 format
encoder_decoder_model = build_encoder_decoder_model(input_shape)
fit_model_with_checkpoint(encoder_decoder_model, (X_train, y_train), (X_val, y_val), os.path.join(save_dir, "encoder_decoder_model"))

print("Training complete.")
