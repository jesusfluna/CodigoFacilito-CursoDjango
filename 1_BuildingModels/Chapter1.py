import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

l0 = tf.keras.Dense(units=1, input_shape=[1])  # Capa con 1 neurona de tipo Dense y una entrada
model = tf.keras.layers.Sequential(l0)  # Se añade la capa al modelo de manera secuencial
model.compile(optimizer='sgd', loss='mean_squared_error')  # Función de optimización y de perdida

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)  # Array de datos para X
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)  # Array de datos para Y

model.fit(xs, ys, epochs=500)  # Entrenamos el modelo con los datos durante 500 épocas (iteraciones)

print(model.predict([10.0]))  # Probamos una predicción para el valor 10
print("Here is what I learned: {}".format(l0.get_weights()))  # pesos de X e Y, cuanto se han acercado a la solución
