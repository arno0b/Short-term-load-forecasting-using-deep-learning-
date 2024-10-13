import tensorflow as tf

def build_encoder_decoder_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        tf.keras.layers.RepeatVector(24),
        tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50, activation='relu')),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), metrics=['RootMeanSquaredError'])
    return model
