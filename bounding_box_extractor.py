import argparse
import logging
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model

from tqdm import tqdm

# we dont want detectron2 loading logs
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("extraction")


def _build_detection_model(model_path, model_zoo_config_file="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"):
    """Builds a dtection model from file path to previous weights and path of model_zoo config file.

    :param model_path: file path to weights file (.pth)
    :param model_zoo_config_file: path in model_zoo, will be ignored if None
    :return: a model wrapped by DefaultPredictor
    """
    cfg = get_cfg()

    if model_zoo_config_file:
        # Get Faster R-CNN model config we started out learning from
        cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_file))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path

    #DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    predictor = DefaultPredictor(cfg)

    logger.debug("Model loaded.")

    return predictor


def _extract_from_image(predictor, image, horizontal_scale=1.5, vertical_scale=1.2, min_width=0, min_height=0, include_bounding_box=False):
    """
    Cuts out all bounding boxes predicted by a predictor from a given image. The box will be scaled equally in all
    directions to include the surroundings.

    :param predictor: the predictor used to detect bounding boxes
    :param image: the source image
    :param horizontal_scale: how much the cutout is scaled horizontally compared to bbox
    :param vertical_scale: how much the cutout is scaled vertically compared to bbox
    :param min_height: the minimum height a bbox has to have to be cut out
    :param min_width: the minimum width a bbox has to have to be cut out
    :param include_bounding_box: whether to include the original bounding box in the cutout
    :return: List of images, which are cutout from image. May be empty.
    """
    outputs = predictor(np.asarray(image))
    instances = outputs["instances"].to("cpu")

    extracted_bounding_boxes = []

    for index, bbox in enumerate(instances.pred_boxes):
        # copy to make sure we do not draw on original image
        working_image = image.copy()
        draw = ImageDraw.Draw(working_image)
        x1, y1, x2, y2 = bbox.tolist()

        if include_bounding_box:
            color = tuple(np.random.choice(range(256), size=3))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        box_width = abs(x1 - x2)
        box_height = abs(y1 - y2)

        # delta to add/subtract to get the coords of our stretched bounding box on both sides
        dx = (box_width * horizontal_scale) // 2
        dy = (box_height * vertical_scale) // 2

        # make sure bounding box doesn't extend beyond actual image and convert to int
        x1 = int(max(0, x1 - dx))
        y1 = int(max(0, y1 - dy))
        x2 = int(min(working_image.size[0], x2 + dx))
        y2 = int(min(working_image.size[1], y2 + dy))

        if box_width < min_width or box_height < min_height:
            logger.warning(f"{input_file} - Skipping box that is too small.")
            continue

        crop = working_image.crop((x1, y1, x2, y2))
        extracted_bounding_boxes.append(crop)

    return extracted_bounding_boxes


if __name__ == "__main__":

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

    args = parser.parse_args()

    if args.verbose >= 3:
        logger.setLevel(logging.DEBUG)
    elif args.verbose >= 2:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 1:
        logger.setLevel(logging.WARNING)

    generated_files = []
    model = _build_detection_model(args.model_path)

    for input_file in tqdm(args.input, disable=(not args.progressbar), leave=None):
        logger.debug(f"Evaluating {input_file}")
        path = pathlib.Path(input_file)
        filename = path.stem
        extension = path.suffix

        im = Image.open(input_file)

        cutouts = _extract_from_image(model, im, include_bounding_box=args.bounding_box)

        if not cutouts:
            logger.info(f"{input_file} - No extractable entities found.")
            continue

        for idx, cutout in enumerate(cutouts):
            outfile = args.output_mask.format(filename=filename, extension=extension[1:], index=idx)
            cutout.save(outfile)
            generated_files.append(outfile)

        logger.info(f"{input_file} - Extracted {len(cutouts)} entities.")

    if args.generated:
        for file_path in generated_files:
            print(file_path)
