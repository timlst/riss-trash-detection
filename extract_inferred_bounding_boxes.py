import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Apply trained model to image and cut out any matches if found.')

parser.add_argument('-i', '--input',
                    required=True,
                    nargs='*',
                    help='filepath of the image to apply the model to.')

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

MIN_WIDTH = 50
MIN_HEIGHT = 50

cfg = get_cfg()
model = build_model(cfg)

# Get Faster R-CNN model config we started out learning from
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = args.model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
print("Model loaded.")

predictor = DefaultPredictor(cfg)

# The scaling of the detected bounding box to the final cutout
HORIZONTAL_SCALE = 1.5
VERTICAL_SCALE = 1.2

for input_file in args.input:

    print(f"Evaluating {input_file}")
    path = pathlib.Path(input_file)
    filename = path.stem
    extension = path.suffix

    original_image = Image.open(input_file)

    outputs = predictor(np.asarray(original_image))
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        print("No extractable entities found.")
        print()
        continue

    extractions = 0
    for index, box in enumerate(instances.pred_boxes):
        # copy to make sure we do not draw on original image
        working_image = original_image.copy()
        draw = ImageDraw.Draw(working_image)
        x1, y1, x2, y2 = box.tolist()

        if args.bounding_box:
            color = tuple(np.random.choice(range(256), size=3))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        box_width = abs(x1 - x2)
        box_height = abs(y1 - y2)

        # delta to add/subtract to get the coords of our stretched bounding box on both sides
        dx = (box_width * HORIZONTAL_SCALE) // 2
        dy = (box_height * VERTICAL_SCALE) // 2

        # make sure bounding box doesn't extend beyond actual image and convert to int
        x1 = int(max(0, x1 - dx))
        y1 = int(max(0, y1 - dy))
        x2 = int(min(working_image.size[0], x2 + dx))
        y2 = int(min(working_image.size[1], y2 + dy))

        if box_width < MIN_WIDTH or box_height < MIN_HEIGHT:
            print("Skipping box that is too small.")
            continue

        # extract box manually, syntax swapped because of numpy things
        cutout = working_image.crop((x1, y1, x2, y2))

        outfile = args.output_mask.format(filename=filename, extension=extension[1:], index=index)
        cutout.save(outfile)
        extractions += 1

    print(f"Extracted {extractions} entities.")
    print()

sys.exit(0)
