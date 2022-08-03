"""
This script loads a classification model, takes an image and gives it a classification (with probabilities if desired)
It also provides the functionality to do classification as a library
"""
import argparse

import torch
from PIL import Image

import warnings

from torchvision import transforms

normalize = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def _image_to_normalized_tensor(im, transform=normalize, device="cuda:0"):
    """Normalizes a given PIL image with a transform and turns it into a tensor on a given device.

    :param im: input image
    :param transform: transform that is applied to image
    :param device: devices where the tensor is stored
    :return: tensor of input image with transform applied
    """
    im = transform(im).float()
    im = im.unsqueeze(0)
    return im.to(device)


def _build_classification_model(model_path):
    """Loads a model from given path to model file and prepares it for inference.
    :param model_path: path to model file (.pth)
    :return: a model in evaluation mode
    """
    model = torch.load(model_path)
    model.eval()

    return model


def _predict_probabilities(classifier, image_tensor):
    """
    Applies a classification model to an image tensor as input and returns a probability distribution over the classes.
    :param classifier: classification model to apply
    :param image_tensor: tensor of input image
    :return: probability distribution over all classes embedded in the classification model
    """
    output = classifier(image_tensor)

    # Softmax normalizes the output to a probability distribution over all classes
    class_probabilities = torch.nn.functional.softmax(output, dim=1)

    return class_probabilities


def _extract_prediction(class_confidence_distribution):
    """
    Given a probability distribution over a number of classes,
    extracts id and probability for prediction with highest confidence
    :param class_confidence_distribution: probability distribution over classes
    :return: tuple of class id and probability of prediction with highest confidence
    """
    # Extract index as well as the highest probability among our predictions
    highest_confidence, likeliest_class = torch.max(class_confidence_distribution, 1)

    return likeliest_class, highest_confidence


def _classify(classifier, image_tensor, class_names, minimal_confidence=0.0):
    """
    Applies a classification model to an image tensor as input and
    returns a tuple of name of predicted class and the corresponding confidence.
    :param classifier: classification model to apply
    :param image_tensor: tensor of input image
    :param minimal_confidence: minimal required confidence for a prediction, otherwise class id will be None
    :return: tuple of class name and confidence, class id may be "Unsure" if the confidence is below minimal_confidence
    """

    predicted_class_id, confidence_ = _extract_prediction(_predict_probabilities(classifier, image_tensor))
    # condense into single value
    confidence_ = confidence_.data[0]

    if confidence_ < minimal_confidence:
        return "Unsure", confidence_

    return class_names[predicted_class_id], confidence_


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Apply classification model to image and predict class.')

    parser.add_argument('-i', '--input',
                        required=True,
                        nargs='*',
                        help='filepath of the images to apply the model to.')

    parser.add_argument('-o', '--output_mask',
                        default='{filename}_out{index}.{extension}',
                        required=False,
                        help='format string for output file, supports keywords "filename", "extension" and "index". '
                             'default: {filename}_out{index}.{extension}')

    parser.add_argument('-m', '--model_path',
                        required=True,
                        help='filepath of the model to use.')

    parser.add_argument('-f', '--full',
                        default=False,
                        action='store_true',
                        help='Show probability for all classes.')

    args = parser.parse_args()

    classifier = _build_classification_model(args.model_path)

    class_names = ("Empty", "Full", "Garbage Bag")

    for input_file in args.input:
        image = Image.open(input_file)
        image_tensor = _image_to_normalized_tensor(image)

        probabilities = _predict_probabilities(classifier, image_tensor)

        print(input_file)
        if args.full:
            for label, probability in zip(class_names, probabilities.squeeze().tolist()):
                print(f"{label}:{probability:.2%}")
        else:
            pred_class, confidence = _classify(classifier, image_tensor, class_names, minimal_confidence=0.5)
            print(pred_class, confidence)
            if not pred_class:
                pred_class = "Unsure"
            print(f"{pred_class}:{confidence:.2%}")
