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

class_names = ["Empty", "Full", "Garbage Bag", ]

CHOSEN_THRESHOLD = 0.85


def imshow(img, title=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()


def apply_confidence_threshold(predictions, confidences, threshold, unsure_symbol=3):
    """
    Take predictions and the confidence for the predictions and produces a new set of predictions, which assigns all
    predictions that fall under the threshold the unsure symbol.
    :param predictions: predicted classes for dataset
    :param confidences:  confidence of predictions
    :param threshold: minimum confidence to be considered "sure"
    :return: adjusted predictions
    """
    return [unsure_symbol if conf < threshold else pred for conf, pred in zip(confidences, predictions)]


def apply_intervention(truths, predictions, confidences, threshold):
    """
    Take truths, predictions and the confidence for the predictions and produces a new set of predictions, which assumes
    a "godly intervention" resorting to ground truth for all predictions under a certain threshold
    :param truths: ground truth classes for dataset
    :param predictions: predicted classes for dataset
    :param confidences:  confidence of predictions
    :param threshold: minimum confidence to be considered "sure"
    :return: (adjusted predictions list, number of adjusted elements)
    """
    assert len(truths) == len(predictions) == len(confidences), "Length of dataset does not match up"
    predictions_unsure = apply_confidence_threshold(predictions, confidences, threshold, unsure_symbol=None)
    interventions = predictions_unsure.count(None)
    return [truth if pred is None else pred for truth, pred in zip(truths, predictions_unsure)], interventions


model = torch.load(args.model_path)
model.eval()

########################
# TEST SET PREDICTIONS #
########################
y_inputs = []
y_confidence = []
y_pred = []
y_true = []

# iterate over test data
for inputs, labels in dataset_loader:
    # Run Inference on GPU if possible
    y_inputs.extend(inputs)
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, likeliest_class = torch.max(probabilities, 1)

    confidence = confidence.data.cpu().numpy()
    likeliest_class = likeliest_class.data.cpu().numpy()
    # Save Prediction
    y_pred.extend(likeliest_class)
    y_confidence.extend(confidence)

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

predictions_unsure = apply_confidence_threshold(y_pred, y_confidence, CHOSEN_THRESHOLD)
predictions_after_intervention, human_count = apply_intervention(y_true, y_pred, y_confidence, CHOSEN_THRESHOLD)


########################
# SHOW FALSE PREDICTIONS
########################

def plot_images_with_labels(images, labels):
    """
    Take list of image tensors (channel first!) and labels, plot as labelled grid
    :param images:
    :param labels:
    """
    ncol = 4
    nrow = math.ceil(len(images) / ncol)
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
            print(class_names[truth])

    plot_images_with_labels(
        # unnormalize images
        images=[(m[0] / 2 + 0.6) for m in mismatches],
        labels=[f"T:{class_names[truth]}|P:{class_names[output]}" for _, output, truth in mismatches]
    )

####################
# CONFUSION MATRICES
####################
os.chdir(args.output)

fig1, ax1, = plt.subplots()
fig2, ax2, = plt.subplots()
sns.set(font="Tahoma")
sns.set(font_scale=1.2)
font = {
    "family" : "Tahoma",
    "size": 16,
}
plt.rc("font", **font)
plt.rcParams['text.color'] = "#4ea6bc"

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
cf_matrix_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

sns.heatmap(cf_matrix_normalized, annot=True, fmt='.2f',
            xticklabels=class_names, yticklabels=class_names,
            cmap=sns.light_palette("#388294", as_cmap=True), ax=ax1)
ax1.set_title(f'Confusion matrix')
ax1.set_ylabel('Ground Truth')
ax1.set_xlabel('Prediction')

# save in output dir
fig1.savefig(f'confusion_matrix_original.png', transparent=True)


# Build confusion matrix under threshold
cf_matrix = confusion_matrix(y_true, predictions_unsure)
cf_matrix_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

# no "Unsure" ground truth row
cf_matrix_normalized_classes_only = np.delete(cf_matrix_normalized, (-1), axis=0)

sns.heatmap(cf_matrix_normalized_classes_only, annot=True, fmt='.2f',
            xticklabels=class_names + ["Unsure", ], yticklabels=class_names,
            cmap=sns.light_palette("#388294", as_cmap=True), ax=ax2)
ax2.set_title(f'Confusion matrix for confidence threshold {CHOSEN_THRESHOLD}')
ax2.set_ylabel('Ground Truth')
ax2.set_xlabel('Prediction')

# save in output dir
fig2.savefig(f'confusion_matrix_{CHOSEN_THRESHOLD}.png', transparent=True)

os.chdir(prev_dir)

#####################
# BALANCED ACCURACY #
#####################
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced accuracy score: {balanced_accuracy}")

print("t,score,interventions")
for t in np.arange(0, 1.1, 0.005):
    predictions_at_threshold, human_inspections = apply_intervention(y_true, y_pred, y_confidence, threshold=t)
    balanced_accuracy_human_intervention = balanced_accuracy_score(y_true, predictions_at_threshold)
    print(f"{t:.4f},{balanced_accuracy_human_intervention:.4f},{human_inspections}")

print("Without human intervention")
precisions, recalls, fbeta_scores, supports = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])
print(f"{'Label': <12}|{'Precision': <10}|{'Recall': <7}|{'Fbeta': <7}|{'Support': <7}")
print('-' * 47)
for label, precision, recall, fbeta, support in zip(class_names, precisions, recalls, fbeta_scores, supports):
    print(f"{label: >12}|{precision: >10.2%}|{recall: >7.2%}|{fbeta: >7.2%}|{support: >7}")
print()

print("With human intervention")
precisions, recalls, fbeta_scores, supports = precision_recall_fscore_support(y_true, predictions_after_intervention,
                                                                              labels=[0, 1, 2])
print(f"{'Label': <12}|{'Precision': <10}|{'Recall': <7}|{'Fbeta': <7}|{'Support': <7}")
print('-' * 47)
for label, precision, recall, fbeta, support in zip(class_names, precisions, recalls, fbeta_scores, supports):
    print(f"{label: >12}|{precision: >10.2%}|{recall: >7.2%}|{fbeta: >7.2%}|{support: >7}")
print()

print(f"Human intervention required for {human_count} images")
# go back to original directory

###################################################################
# "Full" false negative rate at different confidence thresholds
###################################################################
print("t,full_fn_rate")
for t in np.arange(0, 1.1, 0.005):
    predictions_at_intervention, human_inspections = apply_intervention(y_true, y_pred, y_confidence, threshold=t)
    print(predictions_at_intervention)
    fn_rate = predictions_at_intervention.count(1) / y_true.count(1)
    print(f"{t:.4f},{fn_rate:.4f}")

os.chdir(prev_dir)
