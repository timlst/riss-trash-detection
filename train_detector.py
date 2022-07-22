#!/usr/bin/env python3
import os
import sys

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances



class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


cfg = get_cfg()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.merge_from_file("SERVER_DETECTOR.yaml")

# Setup detectron2 logger
setup_logger(output=cfg.OUTPUT_DIR, name="detectron2", abbrev_name="d2")

# dont care, we always train on GPU!
cfg.MODEL.DEVICE = "cuda:0"

WORKING_FOLDER = "./Waste_Bin_Detection_Dataset/combined/"
register_coco_instances("combined_train", {}, f"{WORKING_FOLDER}combined_dataset_train.json", WORKING_FOLDER)
register_coco_instances("combined_validation", {}, f"{WORKING_FOLDER}combined_dataset_validation.json", WORKING_FOLDER)
register_coco_instances("combined_test", {}, f"{WORKING_FOLDER}combined_dataset_test.json", WORKING_FOLDER)

"""
HOW THE CONFIG WAS MADE

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("combined_train",)
cfg.DATASETS.TEST = ("combined_validation",)

# Use SOLVER config from base configuration file
# cfg.SOLVER.BASE_LR = 1e-6  # pick a good LR, wtf is that supposed to mean, previous 0.00025 did not work for R101
# cfg.SOLVER.MAX_ITER = 30000  # arbitrarily chosen
# cfg.SOLVER.STEPS = []  # do not decay learning rate
# cfg.SOLVER.OPTIMIZER = "ADAM"
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people

cfg.DATALOADER.NUM_WORKERS = 8

# Finetune for just finding bounding boxes
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Default may be a good start-
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

with open("SERVER_RESNET101.yaml", "w") as f:
    f.write(cfg.dump())
    """

trainer = MyTrainer(cfg)

print("*" * 10)
print("Starting training.")
print("*" * 10)

trainer.resume_or_load(resume=True)
trainer.train()

print("*" * 10)
print("Starting test set evaluation.")
print("*" * 10)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("combined_test", output_dir=cfg.OUTPUT_DIR)
test_data_loader = build_detection_test_loader(cfg, "combined_test")
print(inference_on_dataset(predictor.model, test_data_loader, evaluator))

# test_dataset_dicts = DatasetCatalog.get("combined_test")
# metadata = MetadataCatalog.get("combined_test")
#
# def show_image_to_user(window_name, im):
#    cv2.imshow(window_name, im)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
# for d in test_dataset_dicts:
#    im = cv2.imread(d["file_name"])
#    outputs = predictor(im)
#    v = Visualizer(im[:, :, ::-1],
#                   metadata=metadata,
#                   scale=1,
#                   instance_mode=ColorMode.IMAGE
#                   )
#    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#    show_image_to_user("prediction", out.get_image()[:, :, ::-1])
#
# print("Inference done")
