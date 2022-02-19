import numpy as np
import tensorflow as tf

X_test = np.array(np.random.random((10, 5)))
Y_test = np.array(np.random.random((10)))

model_restore = tf.saved_model.load("my_model")
print(model_restore)

# res = model_restore.evaluate(X_test, Y_test)
# print("loss and mean_absolute_error", res)
#
# predictions = model_restore.predict(X_test)
# print(predictions)