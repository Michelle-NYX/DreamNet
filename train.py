import os
import coco
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import json
from PIL import Image
from config import Config
import utils
import model as modellib
import visualize
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# local path to mapillary dataset
DATASET_DIR = os.path.join(ROOT_DIR, "mapillary_dataset")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class MapillaryConfig(coco.CocoConfig):
    """Configuration for training on the mapillary dataset.
    Derives from the base Config class and overrides values specific
    to the mapillary dataset.
    """
    # Give the configuration a recognizable name
    NAME = "debug"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    # this MUST be explicitly defined, or will run into index out of bound error
    NUM_CLASSES = 1 + 11  # background + 11 objects

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
#     IMAGE_MIN_DIM = 1024
#     IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0005
    
config = MapillaryConfig()
config.display()

class MapillaryDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    
    DEBUG = False
    CLASS_MAP = {}
    
    CLASSES = ["Bird", "Person", "Bicyclist", "Motorcyclist", "Bench", \
         "Fire Hydrant", "Traffic Light", "Bus", "Car", "Motorcycle", "Truck"]
    
    # local path to image folder, choose 'dev', 'training', or 'testing'
    SUBSET_DIR = ""

    # local path to images inside development folder
    IMG_DIR = ""

    # local path to instance annotations inside development folder
    INS_DIR = ""


    def load_mapillary(self, dataset_dir, subset, class_ids=None,
                  class_map=None):
        
        self.SUBSET_DIR = os.path.join(dataset_dir, subset)
        self.IMG_DIR = os.path.join(self.SUBSET_DIR, 'images')
        self.INS_DIR = os.path.join(self.SUBSET_DIR, 'instances')
        
        # load classes, start with id = 1 to account for background "BG"
        class_id = 1
        for label_id, label in enumerate(class_ids):
            if label["instances"] == True and label["readable"] in self.CLASSES:
                self.CLASS_MAP[label_id] = class_id
                
                if (self.DEBUG):
                    print("{}: Class {} {} added".format(label_id, class_id, label["readable"]))
                    
                self.add_class("mapillary", class_id, label["readable"])
                class_id = class_id + 1
                
        # add images 
        file_names = next(os.walk(self.IMG_DIR))[2]
        for i in range(len(file_names)):
            if file_names[i] != '.DS_Store':
                
                image_path = os.path.join(self.IMG_DIR, file_names[i])
                base_image = Image.open(image_path)
                w, h = base_image.size

                if (self.DEBUG):
                    print("Image {} {} x {} added".format(file_names[i], w, h))

                self.add_image("mapillary", image_id = i,
                              path = file_names[i],
                               width = w,
                               height = h
                              )
                
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        This function loads the image from a file.
        """
        info = self.image_info[image_id]
        img_path = os.path.join(self.IMG_DIR, info["path"])
        image = Image.open(img_path)
        image_array = np.array(image)
        return image_array

    def image_reference(self, image_id):
        """Return the local directory path of the image."""
        info = self.image_info[image_id]
        img_path = os.path.join(self.IMG_DIR, info["path"])
        return img_path

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        instance_path = os.path.join(self.INS_DIR, info["path"])
        instance_image = Image.open(instance_path.rsplit(".", 1)[0] + ".png")
        
        # convert labeled data to numpy arrays for better handling
        instance_array = np.array(instance_image, dtype=np.uint16)

        instances = np.unique(instance_array)
        instaces_count = instances.shape[0]
        
        label_ids = instances // 256
        label_id_count = np.unique(label_ids).shape[0]
        
        if (self.DEBUG):
            print("There are {} instances, {} classes labelled instances in the image {}."\
                  .format(instaces_count, label_id_count, info["path"]))
            
        mask = np.zeros([instance_array.shape[0], instance_array.shape[1], instaces_count], dtype=np.uint8)
        mask_count = 0
        loaded_class_ids = []
        for instance in instances:
            label_id = instance // 256
            if (label_id in self.CLASS_MAP):
                m = np.zeros((instance_array.shape[0], instance_array.shape[1]), dtype=np.uint8)
                m[instance_array == instance] = 1
                m_size = np.count_nonzero(m == 1)
                
                # only load mask greater than threshold size, 
                # otherwise bounding box with area zero causes program to crash
                if m_size > 5000:
                    mask[:, :, mask_count] = m
                    loaded_class_ids.append(self.CLASS_MAP[label_id])
                    mask_count = mask_count + 1
                    if (self.DEBUG):
                        print("Non-zero: {}".format(m_size))
                        print("Mask {} created for instance {} of class {} {}"\
                              .format(mask_count, instance, self.CLASS_MAP[label_id], \
                                      self.class_names[self.CLASS_MAP[label_id]]))
        mask = mask[:, :, 0:mask_count]
        return mask, np.array(loaded_class_ids)

# read in config file
with open(os.path.join(DATASET_DIR, 'config.json')) as config_file:
    class_config = json.load(config_file)
# in this example we are only interested in the labels
labels = class_config['labels']
        
# Training dataset
dataset_train = MapillaryDataset()
dataset_train.load_mapillary(DATASET_DIR, "debug_train", class_ids = labels)
dataset_train.prepare()

# Validation dataset
dataset_val = MapillaryDataset()
dataset_val.load_mapillary(DATASET_DIR, "debug_val", class_ids = labels)
dataset_val.prepare()

print("mapping: ", class_config["mapping"])
print("version: ", class_config["version"])
print("folder_structure:", class_config["folder_structure"])
print("There are {} classes in the config file".format(len(labels)))
print("There are {} classes in the model".format(len(dataset_train.class_names)))
for i in range(len(dataset_train.class_names)):
    print("    Class {}: {}".format(i, dataset_train.class_names[i]))


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
     
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=2, 
            layers='heads')
