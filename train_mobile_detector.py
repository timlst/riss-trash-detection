#!/usr/bin/env python3
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg, LazyConfig
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


cfg = get_cfg()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.merge_from_file("MOBILE_DETECTOR_PAPER2.yaml")

# Setup detectron2 logger
setup_logger(output=cfg.OUTPUT_DIR, name="detectron2", abbrev_name="d2")

# dont care, we always train on GPU!
cfg.MODEL.DEVICE = "cuda:0"

WORKING_FOLDER = "./Waste_Bin_Detection_Dataset/combined/"
register_coco_instances("combined_train", {}, f"{WORKING_FOLDER}combined_dataset_train.json", WORKING_FOLDER)
register_coco_instances("combined_validation", {}, f"{WORKING_FOLDER}combined_dataset_validation.json", WORKING_FOLDER)
register_coco_instances("combined_test", {}, f"{WORKING_FOLDER}combined_dataset_test.json", WORKING_FOLDER)
register_coco_instances("combined_test_tiny", {}, f"{WORKING_FOLDER}combined_dataset_test_tiny.json", WORKING_FOLDER)

"""
HOW THE ORIGINAL CONFIG WAS GENERATED BASED ON

cfg.DATASETS.TRAIN = ("combined_train",)
cfg.DATASETS.TEST = ("combined_validation",)

# Separate this from normal classifier
cfg.OUTPUT_DIR = "output_mobile"

cfg.SOLVER.MAX_ITER = 30000  # after ~30k iters we are mostly stagnant, empirically proven

cfg.SOLVER.IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people

cfg.DATALOADER.NUM_WORKERS = 8
# Train with negative examples
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

# Finetune for just finding bounding boxes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # We are just looking for positives

# Train on GPU
cfg.MODEL.DEVICE = "cuda:0"

# Checkpoint and do Validation once per EPOCH
IMAGES_PER_EPOCH = sum(len(DatasetCatalog.get(training_set)) for training_set in cfg.DATASETS.TRAIN)
ITERATIONS_PER_EPOCH = IMAGES_PER_EPOCH // cfg.SOLVER.IMS_PER_BATCH

print(f"Loaded {IMAGES_PER_EPOCH} images")
print(f"{ITERATIONS_PER_EPOCH} iterations per epoch at batch size of {cfg.SOLVER.IMS_PER_BATCH}")

cfg.TEST.EVAL_PERIOD = ITERATIONS_PER_EPOCH
cfg.SOLVER.CHECKPOINT_PERIOD = ITERATIONS_PER_EPOCH

with open("MOBILE_RESNET18.yaml", "w") as f:
    f.write(cfg.dump())
"""

trainer = MyTrainer(cfg)

delim = '*'*10
print(f"{delim}\nStarting training.\n{delim}")

trainer.resume_or_load(resume=True)
trainer.train()

print(f"{delim}\nStarting test set evaluation.\n{delim}")

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("combined_test", output_dir=cfg.OUTPUT_DIR)
test_data_loader = build_detection_test_loader(cfg, "combined_test")
print(inference_on_dataset(predictor.model, test_data_loader, evaluator))
