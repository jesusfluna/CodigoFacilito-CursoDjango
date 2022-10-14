import tensorflow as tf
data = tf.keras.datasets.fashion_mnist  # data con 60000 imágenes 28x28

(training_images, training_labels), (test_images, test_labels) = data.load_data()   # Dividimos la data en las imagenes y los labels tanto para entrenamiento como para test
training_images = training_images / 255.0  # Normalización de las imágenes (0-1), como son en blanco y negro con valor de 0 a 255
test_images = test_images / 255.0  # Normalización de las imágenes (0-1), como son en blanco y negro con valor de 0 a 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # No es una capa de neuronas, es un "aplanador", convertirá el array2D (imagen 28x28)en una 1D
    keras.layers.Dense(128, activation=tf.nn.relu),  # 1 Capa con función relu, devuelve valores > 0
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 2 Capa con función softmax que devuelve el mayor valor de todas las neuronas (10 porque son 10 los elementos a clasificar)
])

model.compile(optimizer='adam',  # optimizador
loss='sparse_categorical_crossentropy',  # funcion de perdida
metrics=['accuracy'])  # Metrica
model.fit(training_images, training_labels, epochs=5)  # Entrenamiento

model.evaluate(test_images, test_labels)  # Evaluamos el modelo con los datos del test

""" Vemos los valores dado por nuestra predicción con respecto a los reales del test"""
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

"""

# Si entrenamos 50 veces tenemos un 0,96 de precision
model.fit(training_images, training_labels, epochs=50)

# En el test un 0,88 de precision
model.evaluate(test_images, test_labels)

# Vemos que empieza a haber mucha distancia entre la precision del entrenamiento y el test
# esto puede ser un síntoma de sobre entrenamiento.

"""
