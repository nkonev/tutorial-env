import tensorflow as tf

# Назначим два отдельных входа
input_1 = tf.keras.Input(shape=(10,), name='input_1')
input_2 = tf.keras.Input(shape=(20,), name='input_2')

# Определим структуру для обработки первого входа
layer_1 = tf.keras.layers.Dense(32, name='layer_1')(input_1)

# Для второго входа
layer_2 = tf.keras.layers.Dense(32, name='layer_2')(input_2)
layer_3 = tf.keras.layers.Dense(16, name='layer_3')(layer_2)

# Объединим
concatenate = tf.keras.layers.concatenate([layer_1, layer_3])

# Определим два выхода
output_1 = tf.keras.layers.Dense(1, name='output_1')(concatenate)
output_2 = tf.keras.layers.Dense(1, name='output_2')(concatenate)

model_3 = tf.keras.Model(
    inputs=[input_1, input_2],
    outputs=[output_1, output_2],
)

tf.keras.utils.plot_model(model_3, show_shapes=True)
