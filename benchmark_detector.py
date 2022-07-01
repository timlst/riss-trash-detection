import argparse
import logging
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import seaborn as sns
from PIL import Image
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve

from torchvision import transforms
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(description='Generate certain metrics for a given detection model and test dataset')

parser.add_argument('-a', '--annotations',
                    required=True,
                    help='Filepath to COCO annotations.')

parser.add_argument('-i', '--images',
                    required=True,
                    help='Filepath to working folder for images (coco image paths may be relative)')

parser.add_argument('-m', '--model_path',
                    required=True,
                    help='filepath of the model to use.')

parser.add_argument('-o', '--output',
                    required=False,
                    help='Filepath for results',
                    default='./benchmark/')

args = parser.parse_args()

prev_dir = os.getcwd()
os.makedirs(args.output, exist_ok=True)

print("Loading test set.")
register_coco_instances("test_data", {}, args.annotations, args.images)
print("Test set loaded.")

logging.basicConfig(level=logging.INFO)


print("Loading model.")
cfg = get_cfg()
model = build_model(cfg)

# Get Faster R-CNN model config we started out learning from
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = args.model_path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
print("Model loaded.")


CONFIDENCE = 0.5

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("test_data", output_dir=cfg.OUTPUT_DIR)
test_data_loader = build_detection_test_loader(cfg, "test_data")
print(inference_on_dataset(predictor.model, test_data_loader, evaluator))

# go back to original directory
os.chdir(prev_dir)

