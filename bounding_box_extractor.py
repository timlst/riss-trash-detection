"""
This script loads a model based on a given config (and optionally weights) and cuts out all the predicted bounding boxes.
It also provides functionality to crop boxes if only imported as a module.
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
from detectron2.model_zoo import get_config, get_config_file
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt

from tqdm import tqdm

# we dont want detectron2 loading logs
import utils
from detection_model import _build_detection_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extraction")


def _crop_boxes_from_image(image, boxes):
    """
    Given a list of boxes in format (x1, y1, x2, y2), get a list of cutouts of these boxes from a source image.
    :param image: source image
    :param boxes: list of boxes to cut out from source image
    :return: list of images
    """
    cropped_images = []

    for box in boxes:
        working_image = image.copy()
        crop = working_image.crop(box.tolist())
        cropped_images.append(crop)

    return cropped_images


def extract_predictions_from_image(predictor, image, bounding_box_color=None):
    """
    Cuts out all bounding boxes predicted by a predictor from a given image.
    Will draw the original bounding box if given a bounding_box color
    (will draw a list of given instance predictions on an image).

    :param predictor: the predictor used to detect bounding boxes
    :param image: the source image
    :param bounding_box_color: (R, G, B) color of original drawing box, will be ignored if None.
    :return: List of images, which are cutout from image. May be empty.
    """
    predictions = predictor(np.asarray(image))

    if bounding_box_color:
        visualizer = Visualizer(np.asarray(image)[:, :, ::-1])
        out = visualizer.overlay_instances(boxes=predictions["instances"].to("cpu").original_pred_boxes)
        image = Image.fromarray(out.get_image())

    return _crop_boxes_from_image(image, predictions["instances"].pred_boxes)


if __name__ == "__main__":

    # some weird warning in PyTorch, does not concern us
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Apply trained model to image and cut out any matches if found.')

    parser.add_argument('-i', '--input',
                        required=True,
                        nargs='*',
                        help='filepath of the images to apply the model to.')

    parser.add_argument('-o', '--output_mask',
                        default='{filename}_out{index}.{extension}',
                        required=False,
                        help='format string for output file, supports keywords "filename", "extension" and "index". '
                             'default: {filename}_out{index}.{extension}')

    parser.add_argument('-b', '--bounding_box',
                        action='store_true',
                        default=False,
                        help='Include the original bounding box before resizing')

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

    args = parser.parse_args()

    cfg = utils.get_config_from_path(args.config)

    generated_files = []
    model, _ = _build_detection_model(cfg, weights_path=args.weights_path, additional_options=args.additional_options)

    for input_file in tqdm(args.input):
        logger.debug(f"Evaluating {input_file}")
        path = pathlib.Path(input_file)
        filename = path.stem
        extension = path.suffix

        im = Image.open(input_file)

        cutouts = extract_predictions_from_image(model, im,
                                                 bounding_box_color=(255, 0, 0) if args.bounding_box else None)

        if not cutouts:
            logger.info(f"{input_file} - No extractable entities found.")
            continue

        for idx, cutout in enumerate(cutouts):
            outfile = args.output_mask.format(filename=filename, extension=extension[1:], index=idx)
            cutout.save(outfile)
            generated_files.append(outfile)

        logger.info(f"{input_file} - Extracted {len(cutouts)} entities.")
