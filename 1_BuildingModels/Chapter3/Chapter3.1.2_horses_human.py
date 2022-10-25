import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image


def caballo_o_humano(model, imagen):
    x = image.load_img(imagen, target_size=(300, 300))
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    print(classes[0])
    if classes[0] > 0.5:
        print(imagen + " is a human")
    else:
        print(imagen + " is a horse")


"""# Solo Reescalado de las imágenes a 1/255 para normalizarlas
train_datagen = ImageDataGenerator(rescale=1/255)
"""
# Variación de las imágenes para a partir del set actual generar más variaciones y tener más imágenes de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Reescalado de la imagen 1/255
    rotation_range=40,  # Rotaciones de imagen a 40º de izquierda a derecha
    width_shift_range=0.2,  # Desplazamiento de la imagen horizontalmente un 20%
    height_shift_range=0.2,  # Desplazamiento de la imagen verticalmente un 20%
    shear_range=0.2,  # Recorte de un 20%
    zoom_range=0.2,  # Zoom de un 20%
    horizontal_flip=True,  # Aleatoriamente girar la imagen
    fill_mode='nearest'  # Rellenado de pixeles perdidos con los mas cercanos
)

validation_datagen = ImageDataGenerator(rescale=1/255)

training_dir = 'horse-or-human/training/'
validation_dir = 'horse-or-human/validation/'

# Creamos una instancia de ImageDataGenerator nutriéndolo de las imágenes
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),  # Tamaño de las imágenes
    class_mode='binary'  # tipo de clasificacion, como en este caso son dos (humano/caballo) es binaria, si fuesen mas opciones deberia ser "categorical"
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)

# Modelo de la CNN a usar
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),  #definimos 16 filtros 3x3 y la entrada es de imagenes 300x300 a color, por eso el 3 en el input. Params = (3x3x16+16)*3
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Clasificamos entre humano o caballo, con 1 neurona vale (0 una opcion, 1 la segunda)
])

# mostramos el modelo
print(model.summary())

# Modelo
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

"""
# Comentado para no realizar el entrenamiento y usar el modelo almacenado
# Entrenamos la CNN, pero esta vez añadimos por adelantado el bloque de datos de validación
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

# Vamos a hacer un guardado del modelo para no tener que estar entrenando en cada ejecución
model.save('mi_modelo')
"""


# Cargamos el modelo si no queremos realizar el entrenamiento
new_model = tf.keras.models.load_model('mi_modelo')


# Probamos el modelo con algunas imágenes de prueba
caballo_o_humano(new_model, 'Imgs_test/img1.jpg')
caballo_o_humano(new_model, 'Imgs_test/img2.jpg')
caballo_o_humano(new_model, 'Imgs_test/img3.jpg')
caballo_o_humano(new_model, 'Imgs_test/img4.jpg')
caballo_o_humano(new_model, 'Imgs_test/img5.jpg')
caballo_o_humano(new_model, 'Imgs_test/img6.jpg')
caballo_o_humano(new_model, 'Imgs_test/img7.jpg')
caballo_o_humano(new_model, 'Imgs_test/img8.jpg')
caballo_o_humano(new_model, 'Imgs_test/img9.jpg')
caballo_o_humano(new_model, 'Imgs_test/img10.jpg')

