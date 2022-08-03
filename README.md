# RISS Trash Detection
Detecting trash cans and classifying them for my RISS 2022 project.

As part of my participation in the RISS (Robotics Institute Summer Scholar) program at CMU, I work on detectring trash cans and classifying them as empty or full.
The goal is to deploy a pipeline that can use images collected by a camera-equipped traffic bus in the Pittsburgh area to help the bus company in keeping
their streets clean.

This repository provides the code used, but does not include the resulting models for size reasons.

**This repository is a snapshot of my work done during the summer and may or may not be incorrect. I do not guarantee any maintenance in the future.**

You are more then welcome to fork and build upon my code.

## Requirements
There are certain requirements for running the project, which are included in the requirements.txt file.

Detectron2 especially needs some special attention (if you want to use CUDA, which I highly recommend). I refer to the [tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## The pipeline
The current pipeline is a three stage pipeline:
- a FASTER R-CNN based model to do a rough sweep and pre-selection of possible trash can (to run on the bus)
- a Faster R-CNN based model is responsible for detecting and cutting out all trash cans (to run on the server)
- a classifier takes in the cutout and assigns a class: "Full", "Empty" or "Garbage Bag"

### Models
We use models pretrained on the ImageNet data set and only finetune for our purposes.

The bus-side detector is based on Resnet18-FPN available here https://github.com/sxhxliang/detectron2_backbone.
Our training data is based on [this](https://www.kaggle.com/datasets/elirotondo/waste-bin-detection-dataset-riss-2021) set prepared by a previous RISS student. It has modified for our purposes and may be made available later on.

The detector is based on Resnet101-FPN-3x, available in the Detectron2 [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).
Our training data is based on [this](https://www.kaggle.com/datasets/elirotondo/waste-bin-detection-dataset-riss-2021) set prepared by a previous RISS student. It has modified for our purposes and may be made available later on.

The classifier is also based on Resnet101, made available as part of the PyTorch [torchvision.models](https://pytorch.org/vision/stable/models.html) package. The training data has been processed from the same data set given earlier, but with class annotations included. It is available [here](https://www.kaggle.com/datasets/elirotondo/waste-bin-multiclass-detection-dataset-riss2021).

# Overview over code
Every script has its own documentation and most of them provide some kind of CLI, so refer to that for usage. 
Here we just provide a small overview over everything there is:
- Data preparation
  - [sort_images_by_class.py](sort_images_by_class.py)
- Training
  - [train_classifier.py](train_classifier.py)
  - [train_detector.py](train_detector.py)
  - [train_mobile_detector.py](train_mobile_detector.py)
- Inference and using the trained models
  - [detection_model.py](detection_model.py)
  - [bounding_box_extractor.py](bounding_box_extractor.py)
  - [classifier.py](classifier.py)
  - [detect_and_classify.py](detect_and_classify.py)
- Evaluation of existing models
  - [benchmark_classifier.py](benchmark_classifier.py)
  - [benchmark_detector.py](benchmark_detector.py)