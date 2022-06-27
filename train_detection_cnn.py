#!/usr/bin/env python3

# Some basic setup:
# Setup detectron2 logger
from datetime import datetime

from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

# current timestamp as logfile
setup_logger(output="./output/", name="detectron2", abbrev_name="d2")

# import some common libraries
import os, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def show_image_to_user(window_name, im):
    cv2.imshow(window_name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Register waste bin dataset
from detectron2.data.datasets import register_coco_instances

WORKING_FOLDER_CLOUDY = "./Waste_Bin_Detection_Dataset/cloudy/"
WORKING_FOLDER_SUNNY = "./Waste_Bin_Detection_Dataset/sunny/"
# WORKING_FOLDER = "./Waste_Bin_Multi-Class_Detection_Dataset/cloudy/"

register_coco_instances("waste_bin_cloudy_train", {},
                        f"{WORKING_FOLDER_CLOUDY}cloudy_dataset_train.json",
                        WORKING_FOLDER_CLOUDY)
register_coco_instances("waste_bin_cloudy_validation", {},
                        f"{WORKING_FOLDER_CLOUDY}cloudy_dataset_validation.json",
                        WORKING_FOLDER_CLOUDY)
register_coco_instances("waste_bin_cloudy_test", {},
                        f"{WORKING_FOLDER_CLOUDY}cloudy_dataset_test.json",
                        WORKING_FOLDER_CLOUDY)
register_coco_instances("waste_bin_sunny_train", {},
                        f"{WORKING_FOLDER_SUNNY}sunny_dataset_train.json",
                        WORKING_FOLDER_SUNNY)

from detectron2.engine import DefaultTrainer


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


cfg = get_cfg()

# Let training initialize from model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("waste_bin_cloudy_train", "waste_bin_sunny_train")
cfg.DATASETS.TEST = ("waste_bin_cloudy_validation",)

cfg.TEST.EVAL_PERIOD = 1000
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people

cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR, wtf is that supposed to mean
cfg.SOLVER.MAX_ITER = 30000  # arbitrarily chosen
cfg.SOLVER.STEPS = []  # do not decay learning rate

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Default may be a good start-
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Empty, Full, Bag

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

print("*"*10)
print("Training completed.")
print("*"*10)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("waste_bin_cloudy_test")
val_loader = build_detection_test_loader(cfg, "waste_bin_cloudy_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

print("*"*10)
print("Testset evaluation completed.")
print("*"*10)

from detectron2.utils.visualizer import ColorMode
test_dataset_dicts = DatasetCatalog.get("waste_bin_cloudy_test")
metadata = MetadataCatalog.get("waste_bin_cloudy_test")

#for d in random.sample(test_dataset_dicts, 100):
for d in test_dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(
        im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    if not outputs["instances"]:
        continue
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    show_image_to_user("prediction", out.get_image()[:, :, ::-1])

print("Inference done")
