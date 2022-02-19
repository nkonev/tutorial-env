import tensorflow as tf

# model.save("my_model")
model_restore = tf.saved_model.load("my_model")
print(model_restore)

# Можно посмотреть и как модель обучалась.