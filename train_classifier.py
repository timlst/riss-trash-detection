"""
Pretty standard classification training script.
"""
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load our data sets
batch_size = 4

datasets = {
    "train": ImageFolder("./Waste_Bin_Multi-Class_Detection_Dataset_fixed_medium_upwards/combined/train/", transform=data_transforms["train"]),
    "val": ImageFolder("./Waste_Bin_Multi-Class_Detection_Dataset_fixed_medium_upwards/combined/validation/",
                       transform=data_transforms["val"]),
    "test": ImageFolder("./Waste_Bin_Multi-Class_Detection_Dataset_fixed_medium_upwards/combined/test/", transform=data_transforms["test"]),
}

for lbl, folder in datasets.items():
    total_in_folder = len(folder)
    print(f"{lbl}: {total_in_folder} Images")
    for class_name, index in folder.class_to_idx.items():
        class_in_folder = folder.targets.count(index)
        print(f"\t{class_name}: {class_in_folder} | {class_in_folder/total_in_folder:.2%}")

data_loaders = {phase: torch.utils.data.DataLoader(folder, batch_size=16, shuffle=True, num_workers=8)
                for phase, folder in datasets.items()}

dataset_sizes = {phase: len(folder) for phase, folder in datasets.items()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = datasets['train'].classes


def imshow(inp, title=None, ax=None, figsize=(5, 5)):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


# Setting up the model
# load in pretrained and reset final fully connected

model_fe = models.resnet101(pretrained=True)

num_features = model_fe.fc.in_features
# adjust the final layer for the number of output classes
model_fe.fc = nn.Linear(num_features, len(class_names))

base_model = model_fe.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_fe.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    """
  Support function for model training.

  Args:
    model: Model to be trained
    criterion: Optimization criterion (loss)
    optimizer: Optimizer to use for training
    scheduler: Instance of ``torch.optim.lr_scheduler``
    num_epochs: Number of epochs
    device: Device to run the training on. Must be 'cpu' or 'cuda'
  """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, f"output/classifier_medium_{epoch}.pth")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


new_model = train_model(model_fe, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device=device)
new_model.train()
torch.save(new_model, "output/classifier_medium_final.pth")

"""
Run classification on test dataset once to get accuracy
"""
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
test_data_loader = data_loaders["test"]
with torch.no_grad():
    for images, labels in test_data_loader:
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = new_model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {len(test_data_loader.dataset)} test images: {100 * correct / total} %')
