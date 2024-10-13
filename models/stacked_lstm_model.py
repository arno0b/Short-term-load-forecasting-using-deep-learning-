import tensorflow as tf

def build_stacked_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(250, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(150, return_sequences=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), metrics=['RootMeanSquaredError'])
    return model
