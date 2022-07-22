"""
This script loads a model based on a given config (and optionally weights) and outputs all images that have a prediction
over a given confidence threshold
"""
import argparse
import logging
import os.path
import pathlib
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config, get_config_file
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt

from tqdm import tqdm

# we dont want detectron2 loading logs
import utils
from detection_model import _build_detection_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extraction")

if __name__ == "__main__":

    # some weird warning in PyTorch, does not concern us
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Apply trained model to image and output if it predicts a hit.')

    parser.add_argument('-i', '--input',
                        required=True,
                        nargs='*',
                        help='filepath of the images to apply the model to.')

    parser.add_argument('-c', '--config',
                        required=True,
                        help="Config file to be loaded for model. Will look in file system first, then model zoo.")

    parser.add_argument('-w', '--weights_path',
                        required=False,
                        help='filepath of the model weights to use, takes precedence of weights in config.')

    parser.add_argument('-a', '--additional-options',
                        required=False,
                        nargs="+",
                        help="Additional arguments to be passed to the model cfg")

    parser.add_argument('-t', '--confidence-threshold',
                        required=False,
                        default=0.5,
                        help="Minimum confidence to be considered a candidate")

    args = parser.parse_args()

    cfg = utils.get_config_from_path(args.config)

    cfg.MODEL.DEVICE = "cuda:0"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    model, _ = _build_detection_model(cfg, weights_path=args.weights_path, additional_options=args.additional_options, predictor_wrapper=DefaultPredictor)

    seen_images = 0
    candidates = 0

    for idx, input_file in tqdm(enumerate(args.input), disable=False):
        logger.debug(f"Evaluating {input_file}")
        seen_images += 1
        path = pathlib.Path(input_file)

        im = Image.open(input_file)

        prediction = model(np.asarray(im))

        if prediction["instances"]:
            visualizer = Visualizer(np.asarray(im))
            out = visualizer.draw_instance_predictions(prediction["instances"].to("cpu"))
            Image.fromarray(out.get_image()).save(fp=f"{idx}.png")
            print(f"{input_file}")
            candidates += 1

    logger.info("Candidates / Seen Images")
    logger.info(f"{candidates}/{seen_images} {100*candidates/seen_images:.4f}")
