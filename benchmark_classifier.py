import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import seaborn as sns
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve

from torchvision import transforms
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(
    description='Generate certain metrics for a given classification model and test dataset')

parser.add_argument('-i', '--input',
                    required=True,
                    help='Filepath to dataset folder.')

parser.add_argument('-m', '--model_path',
                    required=True,
                    help='filepath of the model to use.')

parser.add_argument('-s', '--show-failures',
                    required=False,
                    action='store_true',
                    default=False,
                    help='Show images where the classifiers prediction is false')

parser.add_argument('-o', '--output',
                    required=False,
                    help='Filepath for results',
                    default='./benchmark/')

args = parser.parse_args()

prev_dir = os.getcwd()
os.makedirs(args.output, exist_ok=True)

# Set up loading of images
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder = ImageFolder(args.input, transform=transform)
dataset_loader = torch.utils.data.DataLoader(folder, batch_size=16, shuffle=True, num_workers=8)

class_names = ("Empty", "Full", "Garbage Bag")


def imshow(img, title=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()


model = torch.load(args.model_path)
model.eval()

CONFIDENCE = 0.5
########################
# TEST SET PREDICTIONS #
########################
y_inputs = []
y_pred = []
y_true = []
img_pred_truth = []

# iterate over test data
for inputs, labels in dataset_loader:
    # Run Inference on GPU if possible
    y_inputs.extend(inputs)
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    predictions = torch.max(probabilities, 1)
    # Save Prediction
    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth


########################
# SHOW FALSE PREDICTIONS
########################

def plot_images_with_labels(images, labels):
    """
    Take list of image tensors (channel first!) and labels, plot as labelled grid
    :param images:
    :param labels:
    """
    nrow = 5
    ncol = math.ceil(len(images) / nrow)
    fig, m_axs = plt.subplots(nrow, ncol, figsize=(32, 32))
    for (image, label, c_ax) in zip(images, labels, m_axs.flatten()):
        c_ax.imshow(image.permute(1, 2, 0))
        c_ax.set_title(label)
        c_ax.axis('off')
    plt.show()


if args.show_failures:
    mismatches = []
    for idx, (output, truth) in enumerate(zip(y_pred, y_true)):
        if output != truth:
            mismatches.append((y_inputs[idx], output, truth))

    plot_images_with_labels(
        # unnormalize images
        images=[(m[0] / 2 + 0.5) for m in mismatches],
        labels=[f"T:{class_names[truth]}|P:{class_names[pred]}" for _, truth, pred in mismatches]
    )

####################
# CONFUSION MATRIX
####################

# constant for image
# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
cf_matrix_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cf_matrix_normalized, annot=True, fmt='.2f',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# save in output dir
os.chdir(args.output)
plt.savefig('confusion_matrix.png')
os.chdir(prev_dir)
print("Confusion matrix generated.")

#####################
# BALANCED ACCURACY #
#####################
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced accuracy score: {balanced_accuracy}")
print()

precisions, recalls, fbeta_scores, supports = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])
print(f"{'Label': <12}|{'Precision': <10}|{'Recall': <7}|{'Fbeta': <7}|{'Support': <7}")
print('-' * 47)
for label, precision, recall, fbeta, support in zip(class_names, precisions, recalls, fbeta_scores, supports):
    print(f"{label: >12}|{precision: >10.2%}|{recall: >7.2%}|{fbeta: >7.2%}|{support: >7}")

# go back to original directory
os.chdir(prev_dir)
