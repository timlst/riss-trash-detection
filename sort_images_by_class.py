import argparse
import json
import os
import sys

import cv2

parser = argparse.ArgumentParser(description="Take an annotation file, cut out annotated regions and sort into folders (will create folders based on categories)")

parser.add_argument('-a', '--annotations',
                    required=True,
                    help="json file with annotations")
# currently does not work
parser.add_argument('-o', '--outdir',
                    help="Path to out directory where the classed images are saved INCLUDING SLASH")

args = parser.parse_args()

HORIZONTAL_SCALE = 1.5
VERTICAL_SCALE = 1.2


def bboxes_intersect(box1, box2):
    """
    Takes in two bounding boxes both as 4-tuples (x, y, width, height) and returns True if they intersect
    """
    return (abs((box1[0] + box1[2] / 2) - (box2[0] + box2[2] / 2)) * 2 < (box1[2] + box2[2])) and (abs((box1[1] + box1[3] / 2) - (box2[1] + box2[3] / 2)) * 2 < (box1[3] + box2[3]))

with open(args.annotations) as coco_file:
    data = json.load(coco_file)

    categories = {c["id"]: c["name"] for c in data["categories"]}

    for name in categories.values():
        os.makedirs(name, exist_ok=True)

    images = {i["id"]: i for i in data["images"]}
    annotations = data["annotations"]

    # build a mapping from image_id to all annotations for that image
    # We may need to consider overlapping annotations for the next step
    image_to_annotations = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
        image_to_annotations[image_id].append(annotation)

    for image_id, annotations in image_to_annotations.items():

        for annotation in annotations:
            category = categories[annotation["category_id"]]
            image = images[image_id]

            original_image = cv2.imread(image["file_name"])
            x1, y1, box_width, box_height = annotation["bbox"]
            x2, y2 = x1 + box_width, y1 + box_height
            # delta to add/subtract to get the coords of our stretched bounding box on both sides
            dx = (box_width * HORIZONTAL_SCALE) // 2
            dy = (box_height * VERTICAL_SCALE) // 2

            # make sure bounding box doesn't extend beyond actual image and convert to int
            x1 = int(max(0, x1 - dx))
            y1 = int(max(0, y1 - dy))
            x2 = int(min(image["width"], x2 + dx))
            y2 = int(min(image["height"], y2 + dy))

            # when considering a Empty/Full trash can annotation it may happen that a garbage bag sits next to it
            # in that case we want to categorize it as garbage bag as well
            if category != "Garbage Bag":
                for other_annotation in annotations:
                    if categories[other_annotation["category_id"]] == "Garbage Bag" and \
                            bboxes_intersect([x1, y1, abs(x2-x1), abs(y2-y1)], other_annotation["bbox"]):
                        # we want to check for intersection with the cut out box, so we have to transform the
                        # coordinates
                        category = "Garbage Bag"
                        break

            # extract box manually, syntax swapped because of numpy things
            cutout = original_image[y1:y2, x1:x2]

            outfile = f"./{category}/{image['id']}_{category}.jpg"
            cv2.imwrite(outfile, cutout)