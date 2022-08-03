"""
Script to visualize/output predictions of a detection model versus the ground truth.
"""
import argparse
import os

import cv2
from PIL import Image
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.colors as mcolors
from tqdm import tqdm

import utils
from detection_model import _build_detection_model

parser = argparse.ArgumentParser(description='Apply trained model to image and draw both predictions and truth.')

parser.add_argument('-c', '--config',
                        required=True,
                        help="Config file to be loaded for model. Will look in file system first, then model zoo.")

parser.add_argument('-w', '--weights_path',
                        required=False,
                        help='filepath of the model weights to use, takes precedence of weights in config.')

parser.add_argument('--skip-groundtruth',
                    required=False,
                    action="store_true",
                    default=False)

parser.add_argument('-t', '--confidence-threshold',
                    required=False,
                    default=0.8,
                    help="Minimum confidence to be considered a candidate")

args = parser.parse_args()
cfg = utils.get_config_from_path(args.config)

cfg.MODEL.DEVICE = "cuda:0"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
model, _ = _build_detection_model(cfg, weights_path=args.weights_path, predictor_wrapper=DefaultPredictor)

WORKING_FOLDER = "./Waste_Bin_Detection_Dataset/combined/"
register_coco_instances("combined_train", {}, f"{WORKING_FOLDER}combined_dataset_train.json", WORKING_FOLDER)
register_coco_instances("combined_validation", {}, f"{WORKING_FOLDER}combined_dataset_validation.json", WORKING_FOLDER)
register_coco_instances("combined_test", {}, f"{WORKING_FOLDER}combined_dataset_test.json", WORKING_FOLDER)


test_dataset_dicts = DatasetCatalog.get("combined_test")
metadata = MetadataCatalog.get("combined_test")

pred_colors = [mcolors.to_rgba("red")] * 100

for idx, d in tqdm(enumerate(test_dataset_dicts)):
    im = cv2.imread(d["file_name"])
    outputs = model(im)

    predicted_boxes = outputs["instances"].pred_boxes
    annotated_boxes = d["annotations"]

    if len(predicted_boxes) < len(annotated_boxes):
        print(f"{idx}.png missing annotation")

    if outputs["instances"].pred_boxes or d["annotations"]:
        v = Visualizer(im[:, :, ::-1],
                  metadata=metadata,
                  scale=1,
                  instance_mode=ColorMode.SEGMENTATION
                  )
        out = v.overlay_instances(boxes=outputs["instances"].to("cpu").pred_boxes, assigned_colors=pred_colors, alpha=0.7) # draw prediction
        # out = v.draw_dataset_dict(d) # draw ground truth, color determined by metadata
        # draw dataset dict manually to control colors etc
        if not args.skip_groundtruth:
            annos = d.get("annotations", None)
            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]
            category_ids = [x["category_id"] for x in annos]
            colors = [mcolors.to_rgb("lime") for x in annos]
            v.overlay_instances(
                boxes=boxes, assigned_colors=colors, alpha=0.7
            )

        fname = os.path.basename(d['file_name'])
        Image.fromarray(out.get_image()).save(f"./demo/{fname}.png")

print("Inference done")
