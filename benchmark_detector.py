import argparse
import logging
import os
import sys

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

parser = argparse.ArgumentParser(description='Generate certain metrics for a given detection model and test dataset')

parser.add_argument('-a', '--annotations',
                    required=True,
                    help='Filepath to COCO ground truth annotations.')

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

parser.add_argument('--skip-evaluation',
                    required=False,
                    default=False,
                    action='store_true',
                    help="Will skip evaluation on the testset and instead just plot PR curve."
                         "Requires coco_instances_results.json in output folder.")

parser.add_argument('-p', '--predictions',
                    required=False,
                    help='Filepath to predictions in COCO format. '
                         'Will be generated in output folder by default, only makes sense with --skip-evaluation',
                    default="./benchmark/coco_instances_results.json")

args = parser.parse_args()

prev_dir = os.getcwd()
os.makedirs(args.output, exist_ok=True)

if not args.skip_evaluation:
    logger.debug("Loading test set.")
    register_coco_instances("test_data", {}, args.annotations, args.images)
    logger.debug("Test set loaded.")


    logger.debug("Loading model.")
    cfg = get_cfg()
    model = build_model(cfg)

    # Get Faster R-CNN model config we started out learning from
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    logger.debug("Model loaded.")

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("test_data", output_dir=args.output)

    test_data_loader = build_detection_test_loader(cfg, "test_data")
    logger.info(inference_on_dataset(predictor.model, test_data_loader, evaluator))

# Generate PR Curve
# Load Ground Truth annotations
logger.debug("Loading ground truth annotations.")
cocoGt = COCO(args.annotations)
logger.debug("Ground truth annotations loaded.")

# load Detections in COCO format from file
logger.debug("Loading detected bounding boxes.")
if not os.path.exists(args.predictions):
    logger.critical("Predictions could not be found. Ending program.")
    sys.exit(1)

cocoDt = cocoGt.loadRes(args.predictions)
logger.debug("Detected bounding boxes loaded.")

print("Running COCO evaluation")

# Technically this is us doing everything twice, but we don't
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.params.imgIds = cocoGt.getImgIds()
cocoEval.evaluate()
cocoEval.accumulate()

precision = cocoEval.eval["precision"]
recall = cocoEval.eval["recall"]

# we don't expect many detections per image, so might as well include the maximum of 100 (index for list [1, 10, 100])
MAX_DETS = 2
# we are interested in boxes regardless of their size, so we choose index 0 for "all" boxes
AREA_INDEX = 0
# we are class-agnostic and therefore have only one category anyways, so that one
CATEGORY_ID = 0

ious = np.arange(0.5, 1, 0.05)
recall = np.arange(0, 1.01, 0.01)

fig, ax = plt.subplots()

for idx, iou in enumerate(ious):
    ax.plot(recall, precision[idx, :, CATEGORY_ID, AREA_INDEX, MAX_DETS], label=f"IoU@{iou:.2f}")

ax.legend(title='PR with different IoUs:')
ax.set_xlabel("Recall")
ax.set_ylabel("Accuracy")

ax.set_title("PR Curve")

plt.show()

# go back to original directory
os.chdir(prev_dir)
