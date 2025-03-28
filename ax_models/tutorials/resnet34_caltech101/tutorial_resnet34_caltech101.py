# Copyright Axelera AI, 2024
# PyTorch Tutorial adapted from
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# We use the standard ResNet34 model from torchvision with the Caltech101 dataset
# Notes: Should NOT use any axelera.app.* imports here, as this is a tutorial
# for showing customers how to port their existing model on Axelera platform

import os
from pathlib import Path
import pickle
import platform
import sys
import tarfile
import time

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

TRANSFORM = {
    'train': transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}
BATCH_SIZE = 16
WEIGHTS = 'resnet34_caltech101.pth'
DATA_ROOT = './data/Caltech101'
VAL_SPLIT = 0.1  # 10% for validation
PATIENCE = 10  # For early stopping
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif platform.system() == 'Darwin':
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def save_indices(indices, filename):
    with open(filename, 'wb') as f:
        pickle.dump(indices, f)


def load_indices(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if x.mode == 'L':
            x = x.convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def download_and_extract_tar_gz(url, extract_to_dir):
    """
    Downloads a .tar.gz file from a given URL, shows the download progress, and extracts it to the
    specified directory, skipping the root folder of the tar file if all files are contained within it.

    :param url: The URL of the .tar.gz file.
    :param extract_to_dir: The directory where the contents of the .tar.gz will be extracted.
    """
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        file_path = os.path.join(extract_to_dir, Path(url).name)
        with open(file_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress = (file.tell() / total_size) * 100
                print(f"Downloaded {progress:.2f}%", end='\r')
        print("\nDownload complete. Extracting files...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to_dir)

        # Remove the downloaded .tar.gz file
        os.remove(file_path)
        print("Extraction complete.")
    else:
        print(f"Failed to download file from {url}. Status code: {response.status_code}")


def train_val_test_loaders(
    transform_train=TRANSFORM['train'],
    transform_val=TRANSFORM['val'],
    root=DATA_ROOT,
    save_splits=True,
):
    # Downloading the dataset from axelera cloud because torchvision.datasets.Caltech101
    # has an issue with downloading the dataset from the original source
    if not os.path.exists(root):
        download_and_extract_tar_gz(
            'https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/Caltech101.tar.gz', root
        )
    full_dataset = datasets.Caltech101(root=root, download=False)

    class_names = full_dataset.categories
    class_file = Path(__file__).parent / "caltech101_classes.txt"
    if not class_file.exists():
        class_file.write_text('\n'.join(class_names) + '\n')

    # Splitting the dataset into train, validation, and test
    num_full = len(full_dataset)
    num_test = int(0.2 * num_full)  # 20% for testing
    num_val = int(0.2 * (num_full - num_test))  # 20% of the remaining for validation
    num_train = num_full - num_test - num_val

    # Checking if we have saved splits already
    indices_root = Path(root) / "indices"
    indices_root.mkdir(exist_ok=True)
    train_indices_path = indices_root / "train_indices.pkl"
    val_indices_path = indices_root / "val_indices.pkl"
    test_indices_path = indices_root / "test_indices.pkl"
    if (
        train_indices_path.exists()
        and val_indices_path.exists()
        and test_indices_path.exists()
        and not save_splits
    ):
        train_indices = load_indices(train_indices_path)
        val_indices = load_indices(val_indices_path)
        test_indices = load_indices(test_indices_path)
    else:
        trainset, valset, testset = random_split(full_dataset, [num_train, num_val, num_test])
        train_indices = trainset.indices
        val_indices = valset.indices
        test_indices = testset.indices

        save_indices(train_indices, train_indices_path)
        save_indices(val_indices, val_indices_path)
        save_indices(test_indices, test_indices_path)

    # wapper to apply the desired transforms
    train_subset = TransformSubset(
        torch.utils.data.Subset(full_dataset, train_indices), transform=transform_train
    )
    val_subset = TransformSubset(
        torch.utils.data.Subset(full_dataset, val_indices), transform=transform_val
    )
    test_subset = TransformSubset(
        torch.utils.data.Subset(full_dataset, test_indices), transform=transform_val
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    return train_loader, val_loader, test_loader, class_names


def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs=25):
    print(f"Device: {DEVICE}")
    since = time.time()
    best_model_params_path = Path(__file__).parent / WEIGHTS
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0
    epochs_no_improve = 0
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(
                dataloaders[phase], desc=f"{phase} epoch {epoch}", leave=False
            ):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                optimizer.step()  # Optimize after every batch during training
            elif phase == 'val':
                scheduler.step(epoch_loss)  # ReduceLROnPlateau step using validation loss

                # Early stopping
                if epoch_acc > best_acc:
                    epochs_no_improve = 0
                    best_acc = epoch_acc
                    print(f"Update best model, best acc: {best_acc:.4f}")
                    torch.save(model.state_dict(), best_model_params_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == PATIENCE:
                        print("Early stopping!")
                        model.load_state_dict(
                            torch.load(best_model_params_path, map_location=DEVICE)
                        )
                        return model

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(torch.load(best_model_params_path, map_location=DEVICE))
    print('Finished Training')
    return model


def build_model(class_names, fixed_feature_extractor=True, exists_weights=None):
    model_conv = models.resnet34(weights='IMAGENET1K_V1')
    if not exists_weights and fixed_feature_extractor:
        for param in model_conv.parameters():
            param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    # TODO: an example to show if modify the model like adding a dropout layer
    # model_conv.fc = nn.Sequential(
    #     nn.Dropout(0.4),
    #     nn.Linear(num_ftrs, len(class_names)),
    # )
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    model_conv = model_conv.to(DEVICE)
    if exists_weights:
        if not exists_weights.is_file():
            raise FileNotFoundError(f"File {exists_weights} does not exist")
        model_conv.load_state_dict(torch.load(exists_weights, map_location=DEVICE))
    return model_conv


def transfer_learning(
    train_loader, val_loader, class_names, fixed_feature_extractor=True, num_epochs=25
):
    # If fixed_feature_extractor = True, reeze the weights for all of the network except that
    # of the final fully connected layer. This last fully connected layer is replaced with a
    # new one with random weights and only this layer is trained. If not, we still initialize
    # the model with pretrained weights to finetune the ConvNet.
    model_conv = build_model(class_names, fixed_feature_extractor)

    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.01, weight_decay=0.0001)
    # Use ReduceLROnPlateau scheduler
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_conv, 'max', factor=0.1, patience=int(PATIENCE / 2), verbose=True
    )
    return train_model(
        train_loader,
        val_loader,
        model_conv,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        num_epochs=num_epochs,
    )


def test(net, testloader, class_names):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader, desc="Measure accuracy on the test dataset"):
            images = data[0].to(DEVICE)
            labels = data[1].to(DEVICE)
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}

    with torch.no_grad():
        for data in tqdm(testloader, desc="Measure accuracy for each class"):
            images = data[0].to(DEVICE)
            labels = data[1].to(DEVICE)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def imshow(inp, title=None):
    """Display image for Tensor."""
    import numpy as np

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, test_loader, class_names, num_images=20):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 4, 4, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\ntrue: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.show()
                    return
        model.train(mode=was_training)


def main(download_only=False, is_test_mode=False):
    try:
        train_loader, val_loader, test_loader, class_names = train_val_test_loaders()
        if download_only:
            return
        if is_test_mode:
            net = build_model(class_names, exists_weights=Path(__file__).parent / WEIGHTS)
        else:
            net = transfer_learning(train_loader, val_loader, class_names, False)
        test(net, test_loader, class_names)
        visualize_model(net, test_loader, class_names)
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except:
        _, evalue, _ = sys.exc_info()
        print(evalue)
        sys.exit(1)


if __name__ == "__main__":
    script_name = Path(sys.argv[0]).name
    if "--help" in sys.argv:
        print(f"Usage: {script_name} [--test] [--download]")
        print("       --test: Use this flag to run in test mode (using pre-trained weights)")
        print("       --download: Use this flag to only download the dataset and exit")
    elif "--download" in sys.argv:
        main(download_only=True)
    elif "--test" in sys.argv:
        main(is_test_mode=True)
    else:
        main()
