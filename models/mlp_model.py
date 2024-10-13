import tensorflow as tf

def build_mlp_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200, activation='relu'), input_shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(150, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50, activation='relu')),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), metrics=['RootMeanSquaredError'])
    return model
