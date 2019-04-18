#coding=utf8
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = './inception_model'
MODEL_FILE = 'classify_image_graph_def.pb'

CACHE_DIR = './tmp/'

INPUT_DATA = 'flower_photos/'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

def create+image_lists(testing_percentage, validation_percentage):
    result = {}
    sub_dirs = {x[0] for x in os.walk(INPUT_DATA)}
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*."+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        label_name = dir_name.lower()

        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images,
        }
    return result

def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
     


