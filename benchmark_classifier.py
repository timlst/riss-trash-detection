import argparse

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve

from torchvision import transforms
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(description='Generate certain metrics for a given classification model and test dataset')

parser.add_argument('-i', '--input',
                    required=True,
                    help='Filepath to dataset folder.')

parser.add_argument('-m', '--model_path',
                    required=True,
                    help='filepath of the model to use.')

parser.add_argument('-o', '--output',
                    required=False,
                    help='Filepath for results',
                    default='./heatmap.png')

args = parser.parse_args()

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

model = torch.load(args.model_path)
model.eval()


########################
# TEST SET PREDICTIONS #
########################
y_pred = []
y_true = []

# iterate over test data
for inputs, labels in dataset_loader:
    # Run Inference on GPU if possible
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)

    # Save Prediction
    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

####################
# CONFUSION MATRIX #
####################

# constant for image
# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
cf_matrix_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cf_matrix_normalized, annot=True, fmt='.2f',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(args.output)
print(f"Confusion matrix saved to {args.output}")
print()

#####################
# BALANCED ACCURACY #
#####################
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced accuracy score: {balanced_accuracy}")

precisions, recalls, fbeta_scores, supports = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])
print(f"{'Label': <12}|{'Precision': <10}|{'Recall': <7}|{'Fbeta': <7}|{'Support': <7}")
print('-'*47)
for label, precision, recall, fbeta, support in zip(class_names, precisions, recalls, fbeta_scores, supports):
    print(f"{label: >12}|{precision: >10.2%}|{recall: >7.2%}|{fbeta: >7.2%}|{support: >7}")

