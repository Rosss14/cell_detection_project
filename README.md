# Detección de núcleos mediante Deep Learning

Este proyecto tiene como objetivo el reentrenamiento mediante fine-tuning de los modelos de detección de objetos, para detectar núcleos de 
células en imágenes histopatológicas.
Se ha utilizado el conjunto de datos _NuCLS_, de imágenes histopatológicas de cáncer de mama, con anotaciones de posición y categoria de núcleos.
Los modelos reentrenados reciben imágenes como entrada, generan detecciones (localización y clasificación) y los devuelven en un diccionario.

![La pagina de inicio](images/ejemplo.png)

El código entero, incluyendo el proceso de procesamiento de los datos, el entrenamiento, la validación y las pruebas de los modelos se pueden encontrar en:
* https://github.com/Rosss14/cell_detection_project/tfg_zavoiko.ipynb
# La estructura del proyecto
    - App/                - El directorio de la aplicación web
        - saved_model/    - El directorio del modelo (en este caso, EfficientDet D0)
        - static/         - Este directorio se utiliza por la aplicación para guardar imágenes
        - templates/      - El directorio de las plantillas HTML del proyecto
        - app.py          - El código del programa, el ejecutable
        - utils.py        - El código para las funciones auxiliares 
        - label_map.pbtxt - fichero de mapa de etiquetas utilizado para inferencia
    
    - Modelos_reentrenados/
        - EfficientDet_D0/
        - EfficientDet_D2/
        - CenterNet_Hourglass-104/
        - SSD_MobileNet_V2_FPNLite/
    
    - label_map.pbtxt     - fichero de mapa de etiquetas (usado para entrenamiento)
    - Utils.py            - contiene las funciones descritas en la memoria de este trabajo
    - README.md

Cada uno de los directorios correspondientes a modelos contienen:

    - checkpoint/         - ficheros de checkpoints, pueden utilizarse como punto de partida para seguir entrenando el modelo
    - saved_model/        - el directorio con ficheros necesarios para cargar y usar el modelo
    - results/            - el directorio de resultados, contiene ficheros JSON de detecciones de cada subconjunto de datos y un .txt con resultados de evaluación de cada clase y cada subconjunto
    - pipeline.config     - el fichero de configuración que se utiliza en entrenamiento, validación y exportación del modelo

# La ejecución rápida
A continuación se puede observar un código que permite usar modelos para hacer inferencia.
En **index** se pasa un valor entre 0 y 3 para elegir el modelo que desea utilizar, el orden se puede observar en la lista **model_dirs**.
En **imagen** se pasa una ruta a imagen histopatológica en la que desea detectar
núcleos.
``` 
import os
import shutil

os.system('git clone https://github.com/Rosss14/cell_detection_project.git')
shutil.move('cell_detection_project/Utils.py', 'Utils.py')

from Utils import *

index = 0       # Un valor entre 0 y 3, para utilizar un modelo u otro (ver la lista siguiente)

model_dirs = ['EfficientDet_D0', 'EfficientDet_D2', 'CenterNet_Hourglass-104', 'SSD_MobileNet_V2_FPNLite']
scores = [0.36, 0.33, 0.29, 0.41] # La lista de scores para el mejor rendimiento de cada modelo

model_dir = model_dirs[index]  ### Elegir el modelo de la lista 
score = scores[index]          ### Elegir el score

model_path = 'cell_detection_project/Modelos_reentrenados/' + model_dir + '/saved_model'

modelo = tf.saved_model.load(model_path)

imagen = ''

imagen_detectada = detectar_nucleos(modelo, imagen, score)
```
El modelo **CenterNet Hourglass-104** pesa demasiado como para incluirlo en el repositorio, por lo que para usar este hay que añadir el siguente código para cargar sus checkpoints desde **kaggle** y exportar el modelo al formato adecuado. Nota que es necesario especificar el nombre de usuario y la clave de kaggle.
```
os.system('kaggle datasets download -d rostyslavzavoiko/train-checkpoints --unzip')
os.system('python models/research/object_detection/exporter_main_v2.py --pipeline_config_path=pipeline.config --trained_checkpoint_dir=training --output_directory=model_exported')
model_path = 'model_exported/saved_model'
                                                   
```
El siguiente notebook sirve para la ejecución rápida, hay que especificar el valor de **index** para elegir el modelo y subir una imagen que desee. Además, es necesario especificar en los campos correspondiente el nombre de usuario y la clave de kaggle, ya que no hacerlo impide la descarga y uso del modelo. Si se usa el **google colab**, se puede subir algunas imágenes :

* https://github.com/Rosss14/cell_detection_project/DEMO.ipynb

# La ejecución de la aplicación web
Para cargar correctamente la aplicación, abrir la carpeta **App/** en Visual Studio Code y ejecutar el script **app.py**.
IMPORTANTE: el programa no funcionará correctamente si se abre cualquier directorio que no sea 'App'.

1. Pulsar el botón **Subir**, elegir una imagen en el ordenador y cargarla.

![La pagina de inicio](images/upload.png)

2. Comprobar que la imagen es la deseada y pulsar el botón **Detectar**. En caso de que se ha equivocado de imagen, pulsar 
**Inicio** en la barra de navegación.

![La pagina de inicio](images/inicio.png)

3. Pasados unos segundos, se podrá visualizar la imagen con detecciones. Pulsar **Inicio** para volver a cargar otra imagen.

![La pagina de inicio](images/detection.png)
