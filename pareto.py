"""
Script that benchmarks the Speed-Accuracy-Tradeoff for a detection by tweaking the Resize that is being applied.
Didn't actually make it into the final paper, so use at your own risk.
"""
import argparse
import json
import logging
import os
import pickle

import torch
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.utils.logger import setup_logger

import utils
from bounding_box_extractor import _build_detection_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pareto")

parser = argparse.ArgumentParser(description='Generate certain metrics for a given detection model and test dataset')


parser.add_argument('-i', '--images',
                    required=True,
                    help='Filepath to working folder for images (coco image paths may be relative)')

parser.add_argument('-c', '--config',
                    required=False,
                    help="File or model zoo path for config file (if you want to evaluate or benchmark)")

parser.add_argument('-w', '--weights_path',
                    required=False,
                    help='filepath of the model weights to use, takes precedence over config.')

parser.add_argument('-a', '--annotations',
                    required=True,
                    help='Filepath to COCO ground truth annotations.')

parser.add_argument('-p', '--predictions',
                    required=False,
                    help='Filepath to predictions in COCO format relative to output folder. '
                         'Will be generated in output folder by default, only makes sense with --skip-evaluation',
                    default="./coco_instances_results.json")

parser.add_argument('-o', '--output',
                    required=False,
                    help='Filepath for results',
                    default='./pareto/')


args = parser.parse_args()

setup_logger(output=args.output)

prev_dir = os.getcwd()
os.makedirs(args.output, exist_ok=True)

logger.debug("Loading test set.")
register_coco_instances("test_data", {}, args.annotations, args.images)
logger.debug("Test set loaded.")

edge_size_to_time = dict()
edge_size_to_mem = dict()
# go from largest to literally smallest possible, under assumption of divisibility by 8
for shortest_edge_size in range(720, 0, -72):
    logger.info(f"Starting size {shortest_edge_size}")
    # its necessary to set threshhold to 0 for COCO to have the full spectrum available (otherwise a default of 0.05
    # is used)
    cfg = utils.get_config_from_path(args.config)
    cfg.MODEL.DEVICE = "cuda:0"
    cfg.INPUT.MIN_SIZE_TEST = shortest_edge_size
    predictor, cfg = _build_detection_model(
        cfg, weights_path=args.weights_path, additional_options=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", "0.0"], predictor_wrapper=DefaultPredictor
    )
    test_data_loader = build_detection_test_loader(cfg, "test_data")
    evaluator = COCOEvaluator("test_data", output_dir=args.output, allow_cached_coco=False)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    inference_on_dataset(predictor.model, test_data_loader, evaluator=evaluator)
    mem = torch.cuda.mem_get_info()
    logger.info(f"Using {mem=} bytes of memory")
    ender.record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    curr_time = starter.elapsed_time(ender)
    edge_size_to_time[shortest_edge_size] = curr_time
    edge_size_to_mem[shortest_edge_size] = mem

    logger.info(f"{shortest_edge_size} done in time {curr_time}ms")

os.chdir(args.output)
with open('timings.dict', 'w') as file:
    file.write(json.dumps(edge_size_to_time))
    file.write(json.dumps(edge_size_to_mem))

os.chdir(prev_dir)