# Detección de núcleos mediante Deep Learning

Este proyecto tiene como objetivo el reentrenamiento mediante fine-tuning de los modelos de detección de objetos, para detectar núcleos de 
células en imágenes histopatológicas.
Se ha utilizado el conjunto de datos NuCLS, de imágenes histopatológicas de cáncer de mama, con anotaciones de posición y categoria de núcleos.
Los modelos reentrenados reciben imágenes como entrada, generan detecciones (localización y clasificación) y los devuelven en un diccionario.

# La estructura del proyecto
    - App/
        - saved_model/
        - static/ 
        - templates/
        - app.py
        - utils.py
        - label_map.pbtxt - fichero de mapa de etiquetas utilizado para inferencia
    
    - Modelos_reentrenados/
        - EfficientDet_D0/
        - EfficientDet_D2/
        - CenterNet_Hourglass-104/
        - SSD_MobileNet_V2_FPNLite/
    
    - label_map.pbtxt     - fichero de mapa de etiquetas (usado para entrenamiento)
    - Utils.py
    - README.md

# La ejecución rápida

``` 
import os
import shutil

os.system('git clone https://github.com/Rosss14/cell_detection_project.git)
shutil.move('cell_detection_project/Utils.py', 'Utils.py')

from Utils import *

model_dirs = ['EfficientDet_D0', 'EfficientDet_D2', 'CenterNet_Hourglass-104', 'SSD_MobileNet_V2_FPNLite']
model_dir = model_dirs[0]  ### Elegir el modelo de la lista 

model_path = 'cell_detection_project/Modelos_reentrenados/' + model_dir + '/saved_model'




```