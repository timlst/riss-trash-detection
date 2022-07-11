import argparse
import logging
import os.path
import pathlib
import sys
import warnings

import numpy as np
from PIL import Image, ImageDraw
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

from tqdm import tqdm

# we dont want detectron2 loading logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extraction")


def _build_detection_model(model_path, config_file="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                           additional_options=None):
    """Builds a detection model from file path to previous weights and path of model_zoo config file.

    :param model_path: file path to weights file (.pth)
    :param config_file: path to config in model_zoo or system, will be ignored if None, will check model zoo first
    :param additional_options: additional cfg parameters that will be passed to the model.
    Always use type appropiate notation regardless of type (i.e 1.0 when float)
    :return: (a model wrapped by DefaultPredictor, config)
    """
    cfg = get_cfg()

    if config_file:
        # Get Faster R-CNN model config we started out learning from
        try:
            # is it a model zoo file?
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except RuntimeError:
            # is it a system file?
            if os.path.exists(config_file):
                cfg.merge_from_file(config_file)

    # so that given weights get loaded by DefaultPredictor
    cfg.MODEL.WEIGHTS = model_path

    if additional_options:
        cfg.merge_from_list(additional_options)

    cfg.freeze()

    # DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    predictor = DefaultPredictor(cfg)

    logger.debug("Model loaded.")

    return predictor, cfg


def _get_bounding_boxes(predictor, image, horizontal_scale=1.5, vertical_scale=1.2, min_width=0, min_height=0,
                        include_original=False):
    """
    Gives a list of all bounding boxes predicted by a predictor from a given image.
    The boxes will be scaled equally in all directions to include the surroundings.
    :param predictor: the predictor used to detect bounding boxes
    :param image: the source image
    :param horizontal_scale: how much the cutout is scaled horizontally compared to bbox
    :param vertical_scale: how much the cutout is scaled vertically compared to bbox
    :param min_height: the minimum height a bbox has to have to be cut out
    :param min_width: the minimum width a bbox has to have to be cut out
    :param include_original: return tuples (scaled_bbox, original_bbox) instead
    :return: List of bounding boxes in image. May be empty.
    """
    outputs = predictor(np.asarray(image))
    instances = outputs["instances"].to("cpu")

    extracted_bounding_boxes = []

    for index, bbox in enumerate(instances.pred_boxes):
        # copy to make sure we do not draw on original image
        x1, y1, x2, y2 = bbox.tolist()

        box_width = abs(x1 - x2)
        box_height = abs(y1 - y2)

        # delta to add/subtract to get the coords of our stretched bounding box on both sides
        dx = (box_width * horizontal_scale) // 2
        dy = (box_height * vertical_scale) // 2

        # make sure bounding box doesn't extend beyond actual image and convert to int
        x1 = int(max(0, x1 - dx))
        y1 = int(max(0, y1 - dy))
        x2 = int(min(image.size[0], x2 + dx))
        y2 = int(min(image.size[1], y2 + dy))

        if box_width < min_width or box_height < min_height:
            logger.warning(f"{input_file} - Skipping box that is too small.")
            continue
        if not include_original:
            extracted_bounding_boxes.append((x1, y1, x2, y2))
        else:
            # this is a bit messy, but I'd like to have the original bbox without doing inference twice
            extracted_bounding_boxes.append(((x1, y1, x2, y2), tuple(bbox.tolist())))

    return extracted_bounding_boxes


def extract_predictions_from_image(predictor, image, horizontal_scale=1.5, vertical_scale=1.2, min_width=0,
                                   min_height=0, bounding_box_color=None):
    """
    Cuts out all bounding boxes predicted by a predictor from a given image. The boxes will be scaled equally in all
    directions to include the surroundings.
    Will draw the original bounding box if given a bounding_box color.
    Wrapper function for _crop_boxes_from_image and _get_bounding_boxes.

    :param predictor: the predictor used to detect bounding boxes
    :param image: the source image
    :param horizontal_scale: how much the cutout is scaled horizontally compared to bbox
    :param vertical_scale: how much the cutout is scaled vertically compared to bbox
    :param min_height: the minimum height a bbox has to have to be cut out
    :param min_width: the minimum width a bbox has to have to be cut out
    :param bounding_box_color: (R, G, B) color of original drawing box, will be ignored if None.
    :return: List of images, which are cutout from image. May be empty.
    """
    bboxes = _get_bounding_boxes(
        predictor, image,
        horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, min_width=min_width, min_height=min_height,
        include_original=True
    )
    # make sure to have a copy, in case we draw on the image
    working_image = image.copy()

    if bounding_box_color:
        # draw original boxes into image
        draw = ImageDraw.Draw(working_image)
        for scaled, original in bboxes:
            draw.rectangle(original, outline=bounding_box_color, width=1)

    return _crop_boxes_from_image(working_image, [scaled for scaled, original in bboxes])


def _crop_boxes_from_image(image, boxes):
    """
    Given a list of boxes in format (x1, y1, x2, y2), get a list of cutouts of these boxes from a source image.
    :param image: source image
    :param boxes: list of boxes to cut out from source image
    :return: list of images
    """
    cutouts = []

    for box in boxes:
        working_image = image.copy()
        crop = working_image.crop(box)
        cutouts.append(crop)

    return cutouts


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

    parser.add_argument('-m', '--model_path',
                        required=True,
                        help='filepath of the model to use.')

    parser.add_argument('-a', '--additional-options',
                        required=False,
                        nargs="+",
                        help="Additional arguments to be passed to the model cfg")

    args = parser.parse_args()


    generated_files = []
    model, _ = _build_detection_model(args.model_path, additional_options=args.additional_options)

    for input_file in tqdm(args.input):
        logger.debug(f"Evaluating {input_file}")
        path = pathlib.Path(input_file)
        filename = path.stem
        extension = path.suffix

        im = Image.open(input_file)

        cutouts = extract_predictions_from_image(model, im, bounding_box_color=(255, 0, 0) if args.bounding_box else None)

        if not cutouts:
            logger.info(f"{input_file} - No extractable entities found.")
            continue

        for idx, cutout in enumerate(cutouts):
            outfile = args.output_mask.format(filename=filename, extension=extension[1:], index=idx)
            cutout.save(outfile)
            generated_files.append(outfile)

        logger.info(f"{input_file} - Extracted {len(cutouts)} entities.")