"""
This script loads a model based on a given config (and optionally weights) and cuts out all the predicted bounding boxes.
It also provides functionality to crop boxes if only imported as a module.
"""
import argparse
import logging
import os.path
import pathlib
import warnings

from PIL import Image, ImageDraw
from detectron2.config import CfgNode, get_cfg
from detectron2.model_zoo import get_config, get_config_file

from tqdm import tqdm

# we dont want detectron2 loading logs
import utils
from detection_model import _get_bounding_boxes, _build_detection_model

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
        crop = working_image.crop(box)
        cropped_images.append(crop)

    return cropped_images


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
