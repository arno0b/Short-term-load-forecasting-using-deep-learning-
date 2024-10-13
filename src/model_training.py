import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def fit_model_with_checkpoint(model, train_data, val_data, model_name):
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f'{model_name}.h5', save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    history = model.fit(
        train_data,
        epochs=100,
        validation_data=val_data,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    return history

def plot_forecast_vs_actual(y_true, y_pred, x_labels, title, save_path='forecast_vs_actual.png'):
    plt.plot(x_labels, y_true, label="Actual")
    plt.plot(x_labels, y_pred, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel('Timestamp')
    plt.ylabel('Load (MW)')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig(save_path)
    plt.show()
