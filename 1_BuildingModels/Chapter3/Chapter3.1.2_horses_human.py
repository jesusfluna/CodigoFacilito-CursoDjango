import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Reescalado de las imágenes a 1/255
train_datagen = ImageDataGenerator(rescale=1/255)
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

# Entrenamos la CNN, pero esta vez añadimos por adelantado el bloque de datos de validación
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

