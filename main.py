import numpy as np
import cv2
import tensorflow
import tensorflow.keras
from absl import logging
from itertools import repeat

yolo_iou_threshold = 0.6  # порог пересечения относительно объединения (iou)
yolo_score_threshold = 0.6
weights_yolov3 = '~/Desktop/kurs4/checkpoints/yolov3.weights'  # путь к файлу весов
weights = '~/Desktop/kurs4/checkpoints/yolov3.tf'  # путь к файлу checkpoint'ов
size = 416             # приводим изображения к этому размеру
checkpoints = '~/Desktop/kurs4/checkpoints/yolov3.tf'
num_classes = 80   # количество классов в модели

YOLO_V3_LAYERS = [
  'yolo_darknet',
  'yolo_conv_0',
  'yolo_output_0',
  'yolo_conv_1',
  'yolo_output_1',
  'yolo_conv_2',
  'yolo_output_2',
]

