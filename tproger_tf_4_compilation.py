import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(5, input_shape=(10,), name='hidden_layer_1'),
  tf.keras.layers.Dense(2, name='output')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Функция потерь
    optimizer='Adam', # Оптимизатор
    metrics=[ # Метрики
        'accuracy', # Если у объекта назначено имя, то можно вызвать объект с его помощью
        tf.keras.metrics.Precision()
    ]
)
