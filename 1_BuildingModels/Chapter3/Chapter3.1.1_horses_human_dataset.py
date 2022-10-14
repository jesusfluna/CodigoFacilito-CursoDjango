import urllib.request
import zipfile

""" Descarga del dataset para entrenar la red neuronal de diferenciación entre humano/caballo"""

# Dirección del archivo con las imágenes de personas/caballos
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"

# # Descargamos el fichero de dataset
# file_name = "horse-or-human.zip"
# training_dir = 'horse-or-human/training/'
# urllib.request.urlretrieve(url, file_name)

# Descargamos el fichero de validación
validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(validation_url, validation_file_name)

# # Lo descomprimimos en el directorio indicado
# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()
