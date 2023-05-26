import os
import random 
import math
import io
import json

from pathlib import Path
from PIL import Image
from shutil import move
from collections import namedtuple
from six import BytesIO

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
from object_detection.utils import visualization_utils as vis_util

import tensorflow as tf
import pandas as pd
import numpy as np

### Particion de datos

def convert_csv(csv_path, img_path):          

    ### Recoger la informacion de todas las etiquetas en cada fichero csv y 
    ### crear un dataframe para todos los datos

    ### csv_path => el directorio de las anotaciones CSV
    ### img_path => el directorio de las imagenes 

    ### Devuelve un dataframe con los datos recogidos de todas las imagenes y 
    ### csv

    label_list = []
    class_list = ['tumor_any', 'sTIL', 'nonTIL_stromal'] # La lista de clases consideradas relevantes
    label_files = os.listdir(csv_path)  
    for raw_csv in label_files:
        csv_file = Path(os.path.join(csv_path, raw_csv))
        
        csv_df = pd.read_csv(csv_file)          # Leer el fichero csv y convertirlo en un dataframe

        num_rel_anns=0                          # Contador para anotaciones de las clases relevantes

        for clase in class_list:
          num_rel_anns += list(csv_df['super_classification']).count(clase)

        if num_rel_anns/len(csv_df['super_classification']) >= 0.5: # Umbral para considerar una imagen
        #if num_rel_anns>0:                                         # Si >50% de las anotaciones pertenecen a clases relevantes
          img_name = raw_csv[0:-3] + 'png'      # El nombre de la imagen correspondiente a etiquetas

          img = np.array(Image.open(os.path.join(img_path, img_name)))
          width = str(img.shape[1])              # El ancho de la imagen
          height = str(img.shape[0])             # La altura de la imagen

          for index, row in csv_df.iterrows():                 #Recorrer todas las anotaciones en el fichero
            if row['super_classification'] in class_list:      #Filtrando las anotaciones (solo clases relevantes)
              label = (img_name,
                       width,
                       height,
                       row['super_classification'],
                       row['xmin'],
                       row['ymin'],
                       row['xmax'],
                       row['ymax']
                       )
              label_list.append(label)
        
    colnames = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] 
    labels_df = pd.DataFrame(label_list, columns = colnames)
    return labels_df

def particion(file_list, val_ratio=0.2, test_ratio=0.1):
  
  ### Dividir la lista de ficheros en tres subconjuntos de datos 
  ### usando proporciones indicadas

  ### file_list => la lista de nombres de ficheros
  ### val_ratio => la fracción de imágenes en el conjunto de validacion
  ### test_ratio => la fracción de imágenes en el conjunto de prueba

  ### Devuelve tres listas, cada una con los nombres de ficheros en 
  ### el respectivo subconjunto de datos

  train_list=[]
  val_list=[]
  test_list=[]

  num_images=len(file_list)
  num_test_images=math.ceil(test_ratio*num_images)  # Tamano del conjunto de prueba
  num_val_images=math.ceil(val_ratio*num_images)    # Tamano del conjunto de validacion

  for i in range(num_test_images):
    idx=random.randint(0, len(file_list)-1)  # Indice al azar
    test_list.append(file_list[idx])         # Agregar a la lista de prueba
    file_list.remove(file_list[idx])         # Eliminar de lista completa

  for i in range(num_val_images):
    idx=random.randint(0, len(file_list)-1)
    val_list.append(file_list[idx])
    file_list.remove(file_list[idx])
  
  train_list=file_list   # Las imágenes que no pertenecen a validación, ni prueba

  return [train_list, val_list, test_list]

def split_data_files(img_path, train_set, val_set, test_set):
  
  ### Crea directorios para cada subconjunto de datos si no existen
  ### y mueve cada imagen del directorio general al subdirectorio
  ### correspondiente a su subconjunto de datos

  ### img_path => el directorio donde se encuentran todas las imágenes
  ### train_set => la lista de nombres de ficheros del conjunto de entrenamiento
  ### val_set => la lista de nombres de imágenes del conjunto de validación
  ### test_set => la lista de nombres de imágenes del conjunto de prueba

  images_train = '/content/train'
  images_val = '/content/val'
  images_test = '/content/test'

  if not os.path.exists(images_train):
    os.makedirs(images_train)
  if not os.path.exists(images_val):
    os.makedirs(images_val)
  if not os.path.exists(images_test):
    os.makedirs(images_test)

  for image in train_set:
    image_path=os.path.join(img_path, image)
    move(image_path, images_train)

  for image in val_set:
    image_path=os.path.join(img_path, image)
    move(image_path, images_val)
  
  for image in test_set:
    image_path=os.path.join(img_path, image)
    move(image_path, images_test)

def split(df, group):
  data = namedtuple('data', ['filename', 'object'])
  gb = df.groupby(group)
  return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

label_map_dict={'tumor_any': 1, 'nonTIL_stromal': 2, 'sTIL': 3}

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

### Pruebas de modelos

def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  return output_dict


def create_GT_json(gt_csv, label_map_path, gt_json, image_id_dict):
  
  ''' parametros:
    gt_csv - el fichero csv de anotaciones juntadas,
             generado en el capitulo 'Procesamiento
             de anotaciones' del presente documento
    label_map_path - la direccion del fichero de 
             mapa de etiquetas.
    gt_json - la dirección del fichero json con 
             las anotaciones en formato nuevo
  '''

  images=[]
  annotations=[]
  categories=[]

  out={}

  # Load the label map and create label dictionary to pass from text class to numeric value
  label_map = label_map_util.load_labelmap(label_map_path)
  label_map_dict = label_map_util.get_label_map_dict(label_map)

  # Read the ground truth CSV file as a pandas DataFrame
  gt_df = pd.read_csv(gt_csv)
  
  # Group by filename to get info about images without repetition
  gt_grouped = {k: v for (k,v) in gt_df.groupby('filename')}


  # Get unique image filenames
  image_list = gt_grouped.keys()
  
  # Fill images[]
  images.extend(
      [
          {
              "file_name": value.iloc[0]['filename'],       ## Select the first row for each group,
              "height": int(value.iloc[0]['height']),       ## and obtain width and height of each image
              "width": int(value.iloc[0]['width']),         ## without iterating through the whole group
              "id": image_id_dict[value.iloc[0]['filename']]
          } for value in gt_grouped.values()
      ]
  )

  # Fill annotations[]
  annotations.extend(
      [
          {
              "image_id": image_id_dict[gt_df.loc[i,'filename']],
              "area": float((gt_df.loc[i, 'xmax'] - gt_df.loc[i, 'xmin'])*(gt_df.loc[i, 'ymax'] - gt_df.loc[i, 'ymin'])),
              "bbox": [
                  int(gt_df.loc[i, 'xmin']),
                  int(gt_df.loc[i, 'ymin']),
                  int(gt_df.loc[i, 'xmax'] - gt_df.loc[i, 'xmin']),
                  int(gt_df.loc[i, 'ymax'] - gt_df.loc[i, 'ymin'])
              ],
              "category_id": label_map_dict[gt_df.loc[i,'class']],
              "iscrowd": 0,
              "id": i+1 ## Assuming the dataframe's rows are indexed with numbers
          }   for i in gt_df.T
      ]
  )

  # Fill categories[]
  categories.extend(
      [
          {
              'id': label_map_dict[k],
              'name': k
          } 
          for k in label_map_dict
      ]
  )

  out['images'] = images
  out['annotations'] = annotations
  out['categories'] = categories

  with open(gt_json, 'w', encoding = 'utf-8') as file:
    json.dump(out, file, ensure_ascii=False)


def create_DT_json(image_dir, model, dt_json, image_id_dict):
  out=[]

  images = os.listdir(image_dir)

  for image in images:
    image_path = os.path.join(image_dir,image)
    image_np=load_image_into_numpy_array(image_path)
    output_dict = run_inference_for_single_image(model, image_np)

    result=[]

    # ID for frame
    image_id=image_id_dict[image]

    img=Image.open(image_path)
    width, height = img.size

    # Translate boxes into COCO format
    boxes = []
    for box in output_dict['detection_boxes']:
      ymin = int(float(box[0]*height))
      xmin = int(float(box[1]*width))
      ymax = int(float(box[2]*height))
      xmax = int(float(box[3]*width)) 

      box_new = []
      box_new.append(xmin)
      box_new.append(ymin)
      box_new.append(xmax-xmin)
      box_new.append(ymax-ymin)

      boxes.append(box_new)
  
    result.extend(
        [
            {
                'image_id': image_id,
                'category_id': int(output_dict['detection_classes'][k]),
                'bbox': box,
                'score': float(output_dict['detection_scores'][k])
            } for k, box in enumerate(boxes)
        ]
    )

    for sample in result:
      out.append(sample)
  


  with open(dt_json, 'w', encoding='utf-8') as file:
    json.dump(out, file, ensure_ascii=False)


def compute_iou(box1, box2):
    # Calculate Intersection over Union (IoU) of two boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    iou = interArea / float(area1 + area2 - interArea)
    
    return iou


def compute_f1(gt_json, dt_json, score_thresh):
    """
    Calcula el F1 score para un umbral de confianza, con las anotaciones verdaderas y las detecciones dadas

    Args:
    - gt_json (JSON file): JSON de anotaciones verdaderas, donde cada anotacion es un diccionario con claves "image_id",
      "area", "bbox", "category_id", "iscrowd", y "id"
    - dt_json (JSON file): JSON de detecciones, cada anotacion es un diccionario con claves "image_id", "category_id",
       "bbox", y "score"

    Devuelve dicionario con claves:
    - 'TP': verdaderos positivos, 
    - 'FP': falsos positivos, 
    - 'FN': falsos negativos, 
    - 'Precision': precision, 
    - 'Recall': recall, 
    - 'F1': f1 score}
    """

    dts = json.load(open(dt_json))
    gts = json.load(open(gt_json))
    gt_annotations = gts['annotations']

    dt_annotations=[]
    for dt_ann in dts:
        if dt_ann['score']>=score_thresh:
            dt_annotations.append(dt_ann)

    # Sort detected annotations by decreasing score
    dt_annotations.sort(key=lambda x: x["score"], reverse=True)

    # Initialize variables
    tp = 0
    fp = 0
    fn = 0
    iou_threshold = 0.5

    # Compute number of true positives and false positives
    for dt_ann in dt_annotations:
        # Find ground truth annotations with the same image id and category id as the detected annotation
        gt_anns = [gt_ann for gt_ann in gt_annotations if gt_ann["image_id"] == dt_ann["image_id"]
                   and gt_ann["category_id"] == dt_ann["category_id"]]

        if not gt_anns:
            # Detected annotation has no corresponding ground truth annotation, so it is a false positive
            fp += 1
        else:
            # Compute iou with all ground truth annotations and find the one with highest iou
            best_iou = 0
            best_gt_ann = None
            for gt_ann in gt_anns:
                iou = compute_iou(dt_ann["bbox"], gt_ann["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_ann = gt_ann

            if best_iou >= iou_threshold:
                # Detected annotation matches a ground truth annotation, so it is a true positive
                tp += 1
                # Remove the matched ground truth annotation from the list so it is not counted again
                gt_annotations.remove(best_gt_ann)
            else:
                # Detected annotation does not match any ground truth annotation, so it is a false positive
                fp += 1

    # Compute number of false negatives
    fn = len(gt_annotations)

    # Compute precision, recall, and f1 score
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}

category_index_test={1: {'id': 1, 'name': 'tumor'},
 2: {'id': 2, 'name': 'stromal'},
 3: {'id': 3, 'name': 'sTIL'}}

def detectar_nucleos(model, image, score_threshold):
  image_np = load_image_into_numpy_array(image)

  output_dict = run_inference_for_single_image(model, image_np)

  image_np_det = np.copy(image_np)

  vis_util.visualize_boxes_and_labels_on_image_array(image_np_det,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index_test,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    min_score_thresh=score_threshold,
    line_thickness=2)
  
  image_det_path = image[:-4] + '_detection.png'

  image_det = Image.fromarray(image_np_det)
  image_det.save(image_det_path)

  return image_det_path