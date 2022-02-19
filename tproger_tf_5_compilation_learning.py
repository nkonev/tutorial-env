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

# Обучим
model.fit(
    X, # Набор входных данных
    Y, # Набор правильных ответов
    validation_split=0.2, # Этот параметр автоматически выделит часть обучающего набора на валидационные данные. В данном случа 20%
    epochs=10, # Процесс обучения завершится после 10 эпох
    batch_size = 8 # Набор данных будет разбит на пакеты (батчи) по 8 элементов набора в каждом.
)
