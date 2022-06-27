import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model

import warnings

from torch import nn

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Apply trained model to image and cut out any matches if found.')

parser.add_argument('-i', '--input',
                    required=True,
                    nargs='*',
                    help='filepath of the image to apply the model to.')

parser.add_argument('-o', '--output_mask',
                    default='{filename}_out{index}.{extension}',
                    required=False,
                    help='format string for output file, supports keywords "filename", "extension" and "index". default: {filename}_out{index}.{extension}')

parser.add_argument('-m', '--model_path',
                    type=argparse.FileType('r'),
                    required=True,
                    help='filepath of the model to use.')

args = parser.parse_args()

cfg = get_cfg()

model = nn.Module()
model.load_state_dict(torch.load(args.model_path))

for input_file in args.input:

    print(f"Evaluating {input_file}")
    path = pathlib.Path(input_file)
    filename = path.stem
    extension = path.suffix

    original_image = cv2.imread(input_file)

    outputs = model(original_image)
    _, predicted = torch.max(outputs.data, 1)

    for index, box in enumerate(instances.pred_boxes):
        # copy to make sure we do not draw on original image
        working_image = original_image.copy()
        x1, y1, x2, y2 = box.tolist()

        if args.bounding_box:
            color = tuple(np.random.choice(range(256), size=3))
            # cv2 does not like my types apparently, so we reaaaally make sure that they are right
            working_image = cv2.rectangle(working_image, (int(x1), int(y1)), (int(x2), int(y2)), (int(color[0]), int(color[1]), int(color[2])), 1)

        box_width = abs(x1 - x2)
        box_height = abs(y1 - y2)

        # delta to add/subtract to get the coords of our stretched bounding box on both sides
        dx = (box_width * HORIZONTAL_SCALE) // 2
        dy = (box_height * VERTICAL_SCALE) // 2

        # make sure bounding box doesn't extend beyond actual image and convert to int
        x1 = int(max(0, x1 - dx))
        y1 = int(max(0, y1 - dy))
        x2 = int(min(working_image.shape[1], x2 + dx))
        y2 = int(min(working_image.shape[0], y2 + dy))

        # extract box manually, syntax swapped because of numpy things
        cutout = working_image[y1:y2, x1:x2]

        outfile = args.output_mask.format(filename=filename, extension=extension[1:], index=index)
        cv2.imwrite(outfile, cutout)

    print(f"Extracted {len(instances)} entities.")
    print()

sys.exit(0)
