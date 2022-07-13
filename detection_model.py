"""Detection model

Provides all necessary utility to load a detection model from config/weights and then use it to extract bounding boxes,
that are scaled to include the surroundings.
"""
import logging

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2_backbone import backbone # to allow custom backbones in cfg file

# we dont want detectron2 loading logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")


class MyPredictor:
    """
    Simple predictor that wraps a model and applies necessary transformations.
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        # load weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def _build_detection_model(base_cfg, weights_path=None,
                           additional_options=None):
    """Builds a detection model from file path to previous weights and path of model_zoo config file.

    :param cfg: ConfigNode for Detectron2
    :param weights_path: file path to weights file (.pth) to be loaded, if nothing given in config
    :param additional_options: additional cfg parameters that will be passed to the model.
    Always use type appropiate notation regardless of type (i.e 1.0 when float)
    :return: (a model wrapped by DefaultPredictor, config)
    """
    cfg = get_cfg()

    cfg.merge_from_other_cfg(base_cfg)

    # so that given weights get loaded by DefaultPredictor, in case they are different from config
    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path

    if additional_options:
        cfg.merge_from_list(additional_options)

    cfg.freeze()

    predictor = MyPredictor(cfg)

    logger.debug("Model loaded.")

    return predictor, cfg


def _get_bounding_boxes(predictor, image, horizontal_scale=1.5, vertical_scale=1.2, min_width=1, min_height=1,
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
            logger.warning(f"Skipping box that is too small.")
            continue
        if not include_original:
            extracted_bounding_boxes.append((x1, y1, x2, y2))
        else:
            # this is a bit messy, but I'd like to have the original bbox without doing inference twice
            extracted_bounding_boxes.append(((x1, y1, x2, y2), tuple(bbox.tolist())))

    return extracted_bounding_boxes
