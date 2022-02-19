# Обучение модели
import tensorflow as tf
import numpy as np

# Инициализируем набор данных случайными цифрами.
X = np.array(np.random.random((100, 5))) # Матрица 100 на 10 с диапазоном значений [0;1]
Y = np.array(np.random.random((100))) # Вектор длины 100 с диапазоном значений [0;1]

# Создадим модель
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

# Скомпилируем
model.compile(
    optimizer='Adam',
    loss='mse',
    metrics=['mean_absolute_error']
)

# Expected output
# print(Y)

# В данном виде модель можно обучить, но гораздо эффективнее это можно сделать, если использовать функционал обратных выходов. С их помощью можно осуществить раннюю остановку обучения для борьбы с переобучением, визуализировать данные и многое другое. Вот пример некоторых из них:

# Если ошибка не уменьшается на протяжении указанного количества эпох, то процесс обучения прерывается и модель инициализируется весами с самым низким показателем параметра "monitor"
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # указывается параметр, по которому осуществляется ранняя остановка. Обычно это функция потреть на валидационном наборе (val_loss)
    patience=2, # количество эпох по истечении которых закончится обучение, если показатели не улучшатся
    mode='min', # указывает, в какую сторону должна быть улучшена ошибка
    restore_best_weights=True # если параметр установлен в true, то по окончании обучения модель будет инициализирована весами с самым низким показателем параметра "monitor"
)

# Сохраняет модель для дальнейшей загрузки
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_model', # путь к папке, где будет сохранена модель
    monitor='val_loss',
    save_best_only=True, # если параметр установлен в true, то сохраняется только лучшая модель
    mode='min'
)

# Сохраняет логи выполнения обучения, которые можно будет посмотреть в специальной среде TensorBoard
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='log', # путь к папке где будут сохранены логи
)

# Обучим
model.fit(
    X,
    Y,
    validation_split=0.2,
    epochs=50,
    batch_size = 8,
    callbacks = [
        early_stopping,
        model_checkpoint,
        tensorboard
    ]
)

# model.save("my_model")
# model_restore = tf.saved_model.load("my_model")