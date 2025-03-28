# Copyright Axelera AI, 2025
# PyTorch Tutorial adapted from
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# We use the standard ResNet34 model from torchvision with the Fruits-360 dataset
# (a dataset of images containing fruits and vegetables
# --- https://github.com/fruits-360/fruits-360-100x100, 2024.08.04 with MIT license)

import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Set the default tensor type to float32
torch.set_default_dtype(torch.float32)

TRANSFORM = {
    'train': transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}
BATCH_SIZE = 128
CACHE_DIR = os.path.join(str(Path.home()), '.cache', 'axelera')
WEIGHTS_DIR = os.path.join(CACHE_DIR, 'weights', 'tutorials')
WEIGHTS = os.path.join(WEIGHTS_DIR, 'resnet34_fruits360.pth')
ONNX_PATH = os.path.join(WEIGHTS_DIR, 'resnet34_fruits360_classifier.onnx')
DATA_ROOT = os.path.join(
    os.getenv('AXELERA_FRAMEWORK', str(Path.home())), 'data', 'fruits-360-100x100'
)

VAL_SPLIT = 0.15  # 15% for validation
PATIENCE = 2  # For early stopping
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


# Clone the repository using subprocess
def clone_repo(repo_url, destination_dir):
    # Check if the destination directory already exists
    if os.path.exists(destination_dir):
        print(f"The directory '{destination_dir}' already exists. Skipping cloning.")
        return

    try:
        # Run the git clone command
        subprocess.run(["git", "clone", repo_url, destination_dir], check=True)
        print(f"Repository cloned successfully into '{destination_dir}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")


def ensure_cache_dirs():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)


def build_dataset(data_dir_path, transform=None):
    return ImageFolder(root=data_dir_path, transform=transform)


def train_val_test_loaders(
    transform_train=TRANSFORM['train'],
    transform_val=TRANSFORM['val'],
    root=DATA_ROOT,
    save_splits=True,
    repo_url='https://github.com/fruits-360/fruits-360-100x100.git',
    destination_dir=DATA_ROOT,
    is_test_mode=False,
):
    print(f"{destination_dir = }")
    clone_repo(repo_url, destination_dir)

    test_dataset = build_dataset(destination_dir + '/Test', transform=None)

    if is_test_mode:
        class_mapping = test_dataset.class_to_idx
        class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]

        test_subset = TransformSubset(test_dataset, transform=transform_val)
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        return None, None, test_loader, class_names

    full_dataset = build_dataset(destination_dir + '/Training', transform=None)

    class_mapping = full_dataset.class_to_idx

    print("******************************************************************************")
    print("Class mapping from dataset:")
    for name, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"{idx:3d}: {name}")
    print("******************************************************************************")

    # Create class_names list in the correct order
    class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]

    # Save to file if it doesn't exist
    class_file = Path(WEIGHTS_DIR) / "fruits360.names"
    if not class_file.exists():
        class_file.write_text('\n'.join(class_names) + '\n')

    # Ensure class indices match between datasets
    assert (
        full_dataset.classes == test_dataset.classes
    ), "Training and Test datasets have different classes!"
    assert full_dataset.classes == class_names, "Dataset classes don't match sorted class names!"

    num_full = len(full_dataset)
    num_val = int(VAL_SPLIT * num_full)  # 15% for validation
    num_train = num_full - num_val

    # Checking if we have saved splits already
    indices_root = Path(root) / "indices"
    indices_root.mkdir(exist_ok=True)
    train_indices_path = indices_root / "train_indices.pkl"
    val_indices_path = indices_root / "val_indices.pkl"

    if train_indices_path.exists() and val_indices_path.exists() and not save_splits:
        train_indices = load_indices(train_indices_path)
        val_indices = load_indices(val_indices_path)
    else:
        trainset, valset = random_split(full_dataset, [num_train, num_val])
        train_indices = trainset.indices
        val_indices = valset.indices

        save_indices(train_indices, train_indices_path)
        save_indices(val_indices, val_indices_path)

    # Create subsets with transforms
    train_subset = TransformSubset(
        torch.utils.data.Subset(full_dataset, train_indices), transform=transform_train
    )
    val_subset = TransformSubset(
        torch.utils.data.Subset(full_dataset, val_indices), transform=transform_val
    )
    test_subset = TransformSubset(test_dataset, transform=transform_val)

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


def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs=2):
    print(f"Device: {DEVICE}")
    model = model.to(DEVICE)
    since = time.time()
    ensure_cache_dirs()
    best_model_params_path = WEIGHTS
    # torch.save(model.state_dict(), best_model_params_path)
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
                # to make it work with 'mps' device
                # inputs = inputs.to(torch.float32).to(DEVICE)
                # labels = labels.to(torch.float32).to(DEVICE)

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

            epoch_loss = float(running_loss / dataset_sizes[phase])
            epoch_acc = float(running_corrects / dataset_sizes[phase])
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                optimizer.step()  # Optimize after every batch during training
            elif phase == 'val':
                scheduler.step(epoch_loss)  # ReduceLROnPlateau step using validation loss
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current learning rate: {current_lr}')

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
                            torch.load(
                                best_model_params_path, map_location=DEVICE, weights_only=True
                            )
                        )
                        return model

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(
        torch.load(best_model_params_path, map_location=DEVICE, weights_only=True)
    )
    print('Finished Training')
    return model


class ResNet34Model(nn.Module):
    def __init__(
        self,
        num_classes=141,
        fixed_feature_extractor=True,
        exists_weights=WEIGHTS,  # Changed to use WEIGHTS constant directly
    ):
        super(ResNet34Model, self).__init__()
        # Load the ResNet34 model with pre-trained weights
        self.model = models.resnet34(weights='IMAGENET1K_V1')
        self.model = self.model.to(DEVICE)

        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes).to(DEVICE)

        # Load existing weights if provided and file exists
        if exists_weights and Path(exists_weights).exists():
            print(f"Loading weights from {exists_weights}")
            state_dict = torch.load(exists_weights, map_location=DEVICE, weights_only=True)

            # Handle both direct state dict and model-wrapped state dict
            if any(k.startswith('model.') for k in state_dict.keys()):
                # Remove the 'model.' prefix from the state_dict keys
                new_state_dict = {
                    k[6:]: v for k, v in state_dict.items() if k.startswith('model.')
                }
            else:
                new_state_dict = state_dict

            self.model.load_state_dict(new_state_dict)
        elif fixed_feature_extractor:
            # Freeze parameters only if no existing weights and fixed_feature_extractor is True
            for param in self.model.parameters():
                param.requires_grad = False

        self.model = self.model.to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        return self.model(x)


def transfer_learning(
    train_loader, val_loader, class_names, fixed_feature_extractor=True, num_epochs=6
):
    # If fixed_feature_extractor = True, reeze the weights for all of the network except that
    # of the final fully connected layer. This last fully connected layer is replaced with a
    # new one with random weights and only this layer is trained. If not, we still initialize
    # the model with pretrained weights to finetune the ConvNet.
    # model_conv = build_model(len(class_names), fixed_feature_extractor)
    model_conv = ResNet34Model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.Adam(model_conv.model.fc.parameters(), lr=0.01, weight_decay=0.0001)
    # Use ReduceLROnPlateau scheduler
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_conv, mode='max', factor=0.1, patience=int(PATIENCE / 2)
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
    net = net.to(DEVICE)
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
    model = model.to(DEVICE)
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 12))

    # Create a shuffled loader just for visualization
    shuffled_loader = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=True,  # Enable shuffling here
        num_workers=test_loader.num_workers,
    )

    # Create a list to store all samples
    all_samples = []

    with torch.no_grad():
        for inputs, labels in shuffled_loader:  # Use shuffled loader
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            all_samples.extend(list(zip(inputs, labels)))
            if len(all_samples) >= num_images:  # Stop once we have enough samples
                break

        # Randomly sample from collected samples
        import random

        selected_samples = random.sample(all_samples, min(num_images, len(all_samples)))

        for inputs, labels in selected_samples:
            images_so_far += 1
            outputs = model(inputs.unsqueeze(0))
            _, preds = torch.max(outputs, 1)

            ax = plt.subplot(num_images // 4, 4, images_so_far)
            ax.axis('off')
            pred_idx = preds[0].item()
            label_idx = labels.item()
            ax.set_title(f'predicted: {class_names[pred_idx]}\ntrue: {class_names[label_idx]}')
            imshow(inputs.cpu().data)

            if images_so_far == num_images:
                break

    model.train(mode=was_training)
    plt.tight_layout()
    plt.show()


def export_to_onnx(model, input_tensor=None, onnx_file_path=None, opset_version=17):
    """
    Export a PyTorch model to ONNX format.

    Parameters:
    - model: The PyTorch model to be exported.
    - input_tensor: A sample input tensor that matches the input shape the model expects.
    - onnx_file_path: File path to save the ONNX model (e.g., 'model.onnx').
    - opset_version: The ONNX version to export the model to. Default is 11.
    """
    ensure_cache_dirs()
    model.eval()

    input_tensor = (
        torch.randn(1, 3, 100, 100, device='cpu') if input_tensor is None else input_tensor
    )
    onnx_file_path = ONNX_PATH if onnx_file_path is None else onnx_file_path

    # Export the model
    torch.onnx.export(
        model,  # Model to export
        input_tensor,  # A sample input tensor
        onnx_file_path,  # File path where the ONNX model will be saved
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=opset_version,  # Specify the ONNX opset version
        do_constant_folding=True,  # Whether to perform constant folding for optimization
        input_names=['input'],  # Name for input node(s)
        output_names=['output'],  # Name for output node(s)
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },  # Dynamic axis for batching
    )
    print(f"Model exported to: {onnx_file_path}")


def main(download_only=False, is_test_mode=False):
    try:
        train_loader, val_loader, test_loader, class_names = train_val_test_loaders()
        if download_only:
            return
        if is_test_mode:
            net = ResNet34Model(len(class_names), fixed_feature_extractor=False).eval()
        else:
            net = transfer_learning(train_loader, val_loader, class_names, True, num_epochs=1)
        test(net, test_loader, class_names)
        visualize_model(net, test_loader, class_names)
        if not is_test_mode:
            export_to_onnx(net)
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
