from tensorflow.keras.applications.inception_v3 import InceptionV3
import urllib.request

weights_url = "https://storage.googleapis.com/mledudatasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)  # Descargamos una serie de capas convolutivas del paquete

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),  # Creamos un modelo
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(weights_file)  # Cargamos el modelo con las capas descargadas previamente

pre_trained_model.summary()  # Estructura del modelo
