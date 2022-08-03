"""
Detection model

Provides all necessary utility to load a detection model from config/weights and then use it to extract bounding boxes,
that are scaled to include the surroundings by providing a Predictor that can wrap a loaded model.
"""
import logging

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.structures import Boxes
# to allow custom backbones in cfg file from here https://github.com/sxhxliang/detectron2_backbone
from detectron2_backbone import backbone

# we don't want Detectron2 loading logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")


class ScaledBoundingBoxPredictor:
    """
    Predictor that wraps a model and scales predicted bounding boxes in all directions by a predetermined factor.
    Adds an "original_pred_boxes" field to every instances object.
    Filters out bounding boxes that do not have a minimum area.
    """

    def __init__(self, cfg, horizontal_scale=1.5, vertical_scale=1.2, min_area=1):
        self.cfg = cfg.clone()  # cfg can be modified by model

        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.min_area = min_area

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
            predictions = self._scale_boxes(predictions)
            predictions = self._filter_boxes(predictions)
            print(predictions)

            return predictions

    def _scale_boxes(self, predictions):
        """Scales all predicted bounding boxes by given scale factors."""
        instances = predictions["instances"]
        if instances.pred_boxes:
            # save for visualization purposes
            instances.original_pred_boxes = instances.pred_boxes.clone()

            def scale_single_box(tensor):
                # i am sure there is some Matrix/Tensor mathmagic that does all this but in like a line of code
                x1, y1, x2, y2 = tensor.tolist()

                box_width = abs(x1 - x2)
                box_height = abs(y1 - y2)

                # delta to add/subtract to get the coords of our stretched bounding box on both sides
                dx = (box_width * self.horizontal_scale) // 2
                dy = (box_height * self.vertical_scale) // 2

                # adjust the original tensor, so we preserve shape and location
                tensor[0] = x1 - dx
                tensor[1] = y1 - dy
                tensor[2] = x2 + dx
                tensor[3] = y2 + dy

                return tensor
            # iterate over boxes as (4,) tensors, scale them appropiately and stack back to (Nx4) tensor - guh
            instances.pred_boxes = Boxes(torch.stack(tuple(map(scale_single_box, instances.pred_boxes))))
            # the scaled boxes may go out of bounds which we do not want
            instances.pred_boxes.clip(instances.image_size)


        return predictions

    def _filter_boxes(self, predictions):
        """Filters out bounding boxes smaller than a minimal area."""
        instances = predictions["instances"]
        if instances.pred_boxes:
            mask = instances.original_pred_boxes.area() > self.min_area
            instances.pred_boxes = instances.pred_boxes[mask]
            instances.original_pred_boxes = instances.original_pred_boxes[mask]
        return predictions


def _build_detection_model(base_cfg, weights_path=None,
                           additional_options=None, predictor_wrapper=ScaledBoundingBoxPredictor):
    """Builds a detection model from file path to previous weights and path of model_zoo config file.

    :param cfg: ConfigNode for Detectron2
    :param weights_path: file path to weights file (.pth) to be loaded, if nothing given in config
    :param additional_options: additional cfg parameters that will be passed to the model.
    Always use type appropiate notation regardless of type (i.e 1.0 when float)
    :param predictor_wrapper: A Predictor wrapper class that will be applied to the model
    :return: (a model wrapped by predictor_wrapper, config)
    """
    cfg = get_cfg()

    cfg.merge_from_other_cfg(base_cfg)

    # so that given weights get loaded by DefaultPredictor, in case they are different from config
    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path

    if additional_options:
        cfg.merge_from_list(additional_options)

    cfg.freeze()

    predictor = predictor_wrapper(cfg)

    logger.debug("Model loaded.")

    return predictor, cfg
