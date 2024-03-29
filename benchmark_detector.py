"""
A script that generates certain metrics for a given detection model and test dataset (in COCO format)
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import trange

# in case we want to use RESNET18 or some other non-default backbones
# reguires this: https://github.com/sxhxliang/detectron2_backbone
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config

import utils
from bounding_box_extractor import _build_detection_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

parser = argparse.ArgumentParser(description='Generate certain metrics for a given detection model and test dataset')


parser.add_argument('-i', '--images',
                    required=True,
                    help='Filepath to working folder for images (coco image paths may be relative)')

parser.add_argument('-c', '--config',
                    required=False,
                    help="File or model zoo path for config file (if you want to evaluate or benchmark)")

parser.add_argument('-w', '--weights_path',
                    required=False,
                    help='filepath of the model weights to use, takes precedence over config.')

parser.add_argument('-a', '--annotations',
                    required=True,
                    help='Filepath to COCO ground truth annotations.')

parser.add_argument('-p', '--predictions',
                    required=False,
                    help='Filepath to predictions in COCO format relative to output folder. '
                         'Will be generated in output folder by default, only makes sense with --skip-evaluation',
                    default="./coco_instances_results.json")

parser.add_argument('--skip-evaluation',
                    required=False,
                    default=False,
                    action='store_true',
                    help="Will skip evaluation on the testset and instead just plot PR curve."
                         "Requires coco_instances_results.json in output folder.")

parser.add_argument('--skip-plot',
                    required=False,
                    default=False,
                    action='store_true',
                    help="Will skip the plot"
                    )

# parser.add_argument('--run-benchmark',
#                     required=False,
#                     default=False,
#                     action='store_true',
#                     help="Will test inference time on the testset.")

parser.add_argument('-o', '--output',
                    required=False,
                    help='Filepath for results',
                    default='./benchmark/')

args = parser.parse_args()

# generate output directory and change back to original dir at the end
prev_dir = os.getcwd()
os.makedirs(args.output, exist_ok=True)

########################################################################################################################
# INITIALIZE MODEL
# loads the detection model in case we will need it
########################################################################################################################


if not args.skip_evaluation or args.run_benchmark:
    # only initialize model and load data if required by one of the two
    logger.debug("Loading test set.")
    register_coco_instances("test_data", {}, args.annotations, args.images)
    logger.debug("Test set loaded.")

    logger.debug("Building model.")
    # it's necessary to set threshold to 0 for COCO to have the full spectrum available (otherwise a default of 0.05
    # is used, which introduces a little drop at the edge)
    cfg = utils.get_config_from_path(args.config)
    cfg.MODEL.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor, cfg = _build_detection_model(
        cfg, weights_path=args.weights_path, additional_options=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", "0.0"]
    )
    print(cfg)
    logger.debug("Model loaded.")

    test_data_loader = build_detection_test_loader(cfg, "test_data")

########################################################################################################################
# REDO EVALUATIONS ON TEST SET
# if we aren't skipping evaluation (default), then we want to redo the inference on the test data set
########################################################################################################################
if not args.skip_evaluation:
    evaluator = COCOEvaluator("test_data", output_dir=args.output)

    logger.info(inference_on_dataset(predictor.model, test_data_loader, evaluator))


########################################################################################################################
# PRECISION RECALL CURVE
# generates PR curve from the predictions file (either given as argument or generated in the previous step)
########################################################################################################################
# Generate PR Curve
# Load Ground Truth annotations
logger.debug("Loading ground truth annotations.")
cocoGt = COCO(args.annotations)
logger.debug("Ground truth annotations loaded.")

# load Detections in COCO format from file
logger.debug("Loading detected bounding boxes.")
os.chdir(args.output)
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
if not args.skip_plot:
    # we don't expect many detections per image, so might as well include the maximum of 100 (index for list [1, 10, 100])
    MAX_DETS = 2
    # we are class-agnostic and therefore have only one category anyways, so that one
    CATEGORY_ID = 0

    # what IoU steps are we interested in?
    ious = np.arange(0.5, 0.51, 0.05)

    # what bounding box sizes should be considered? (give as indices for the list below)
    areas = [0, 3, ]
    area_labels = ["all", "small", "medium", "large"]

    recall = np.arange(0, 1.01, 0.01)

    fig, ax = plt.subplots()

    for idx, iou in enumerate(ious):
        for area_index in areas:
            ax.plot(recall, precision[idx, :, CATEGORY_ID, area_index, MAX_DETS], label=f"IoU@{iou:.2f} for {area_labels[area_index]} areas")
    for line in ax.lines:
        print(line.get_label())
        print(line.get_xdata(), line.get_ydata())

    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.5)

    ax.set_title("PR Curve")

    plt.show()

########################################################################################################################
# CALCULATE AP50s/m/l
# not part of standard COCO output, so we calculate them manually
########################################################################################################################
print("AP50sml")
for area in [1, 2, 3]:
    print(f"AP50{area_labels[area]}", np.average(precision[0, :, CATEGORY_ID, area, MAX_DETS]))


########################################################################################################################
# HIGHLIGHT FALSE NEGATIVES
# outputs all images where we NEVER find a bounding box in regardless of IoU
########################################################################################################################

uniques = set()
for img in cocoEval.evalImgs:
    if 0 in img["gtMatches"][0]:
        c = np.count_nonzero(img["gtMatches"][0] == 0)
        uniques.add((img["image_id"], f"{c} Ground truths with no DT match"))
for image_id, s in uniques:
    print(s, cocoGt.imgs.get(image_id)["file_name"])


# Benchmark Inference time over multiple runs - takes forever, there has to be a better way to do this

# def benchmark_inference_time(pred: DefaultPredictor, data, device):
#     dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#     repetitions = 10
#     timings = np.zeros((repetitions, 1))
#
#     # MEASURE PERFORMANCE
#     with torch.no_grad():
#         for rep in trange(repetitions):
#             starter.record()
#             inference_on_dataset(pred.model, data, evaluator=None)
#             ender.record()
#             # WAIT FOR GPU SYNC
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)
#             timings[rep] = curr_time
#
#     # Discard first result as they are warmup
#     timings = timings[1:]
#
#     mean_syn = np.sum(timings) / repetitions
#     std_syn = np.std(timings)
#     print(mean_syn)
#     print(std_syn)
#
#
# if args.run_benchmark:
#     print("Starting benchmark.")
#     benchmark_inference_time(predictor, test_data_loader, device="cuda:0")

# go back to original directory
os.chdir(prev_dir)
