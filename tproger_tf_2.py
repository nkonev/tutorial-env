import tensorflow as tf

input = tf.keras.Input(shape=(10,), name='input')
layer_1 = tf.keras.layers.Dense(32, activation='relu', name='hidden_layer_1')(input)
dropout = tf.keras.layers.Dropout(0.2, name='dropout')(layer_1)
layer_2 = tf.keras.layers.Dense(10, name='hidden_layer_2')(dropout)
layer_3 = tf.keras.layers.Dense(5, activation='relu', name='hidden_layer_3')(layer_2)
output = tf.keras.layers.Dense(2, name='output')(layer_3)

model_2 = tf.keras.Model(
    inputs=input,
    outputs=output,
)

model_2.summary()


tf.keras.utils.plot_model(model_2, show_shapes=True)
