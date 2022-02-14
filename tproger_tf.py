import tensorflow as tf

# Проще всего создать модель с помощью класса Sequential, который принимает список слоев.
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(32, activation='relu', input_shape=(10,), name='hidden_layer_1'),
  tf.keras.layers.Dropout(0.2, name='dropout'),
  tf.keras.layers.Dense(10, name='hidden_layer_2')
])
model.summary()
