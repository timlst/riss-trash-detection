import argparse
import pathlib
import sys

import cv2
import torch
from PIL import Image
from detectron2.config import get_cfg

import warnings

from torch import nn
from torchvision import models
from torchvision import transforms

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Apply trained model to image and cut out any matches if found.')

parser.add_argument('-i', '--input',
                    required=True,
                    nargs='*',
                    help='filepath of the image to apply the model to.')

parser.add_argument('-o', '--output_mask',
                    default='{filename}_out{index}.{extension}',
                    required=False,
                    help='format string for output file, supports keywords "filename", "extension" and "index". default: {filename}_out{index}.{extension}')

parser.add_argument('-m', '--model_path',
                    required=True,
                    help='filepath of the model to use.')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def image_loader(image_name):
    """load image apply transformation and return a CUDA tensor to apply the model to."""
    im = Image.open(image_name)
    im = transform(im).float()
    im = im.unsqueeze(0)
    return im.cuda()


# the model's probability for a class must be at least this high, otherwise the answer is ambiguous
MINIMUM_CONFIDENCE = 0.5


classes = ("Empty", "Full", "Garbage Bag")

args = parser.parse_args()

cfg = get_cfg()

model = torch.load(args.model_path)
model.eval()

for input_file in args.input:
    path = pathlib.Path(input_file)
    filename = path.stem
    extension = path.suffix

    # We have to apply the transform to images first, DatasetLoader does not do this for us here
    image = image_loader(input_file)
    outputs = model(image)
    # Softmax normalizes the tensor to a probability distribution for our 3 classes
    class_probabilities = torch.nn.functional.softmax(outputs, dim=1)
    # Extract index as well as the highest probability among our predictions
    highest_probability, index = torch.max(class_probabilities, 1)

    if highest_probability < MINIMUM_CONFIDENCE:
        print("Image is ambiguous")
        for label, probability in zip(classes, class_probabilities.squeeze().tolist()):
            print(f"{label}:{probability:.2%}")
    else:
        # extract item so format gets a float value
        print(f"{classes[index]}:{highest_probability.item():.2%}")

sys.exit(0)
