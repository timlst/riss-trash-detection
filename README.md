# RISS Trash Detection
Detecting trash cans and classifying them for my RISS 2022 project.

As part of my participation in the RISS (Robotics Institute Summer Scholar) program at CMU, I work on detectring trash cans and classifying them as empty or full.
The goal is to deploy a pipeline that can use images collected by a camera-equipped traffic bus in the Pittsburgh area to help the bus company in keeping
their streets clean.

This repository provides a snapshot of the code used, but does not include the resulting models for size reasons.

## Requirements
There are certain requirements for running the project, which are included in the requirements.txt file.

Detectron2 especially needs some special attention (if you want to use CUDA, which I highly recommend). I refer to the [tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## The pipeline
The current pipeline is a two stage pipeline:
- a Faster R-CNN based model is responsible for cutting out potentially interesting trash cans
- a classifier takes in the cutout and assigns a class: "Full", "Empty" or "Garbage Bag"
