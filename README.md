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
The current pipeline is a two stage pipeline:
- a Faster R-CNN based model is responsible for detecting and cutting out potentially interesting trash cans
- a classifier takes in the cutout and assigns a class: "Full", "Empty" or "Garbage Bag"

### Models
We use models pretrained on the ImageNet data set and only finetune for our purposes.

The detector is based on Resnet101-FPN-3x, available in the Detectron2 [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).
Our training data is based on [this](https://www.kaggle.com/datasets/elirotondo/waste-bin-detection-dataset-riss-2021) set prepared by a previous RISS student. It has modified for our purposes and may be made available later on.

The classifier is also based on Resnet101, made available as part of the PyTorch [torchvision.models](https://pytorch.org/vision/stable/models.html) package. The training data has been processed from the same data set given earlier, but with class annotations included. It is available [here](https://www.kaggle.com/datasets/elirotondo/waste-bin-multiclass-detection-dataset-riss2021).

# Usage
Both detector/extractor and classifier can be used separately or can be imported for further use. To see an example of this see [here](detect_and_classify.py).

Furthermore all scripts are runnable and provide a CLI to explain different parameters.
