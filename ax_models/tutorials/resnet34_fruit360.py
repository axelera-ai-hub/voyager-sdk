# Copyright Axelera AI, 2023
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

# Set up device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif platform.system() == 'Darwin':
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Set default device and data type handling
def to_device(tensor, device=None, dtype=None):
    """Move tensor to the specified device and ensure correct dtype.

    Args:
        tensor: The tensor to move
        device: Target device (defaults to global DEVICE)
        dtype: Target data type (defaults to float32 for floating point tensors)

    Returns:
        The tensor on the target device with the correct data type
    """
    if device is None:
        device = DEVICE

    # Handle data type conversion for floating point tensors
    if dtype is None and tensor.is_floating_point():
        dtype = torch.float32

    # Move tensor to device and convert type if needed
    return tensor.to(device=device, dtype=dtype)


# Default commit hash for the dataset that matches the pre-trained weights (141 classes)
DEFAULT_COMMIT_HASH = "2f981c83e352a9d4c15fb8c886034c817052c80b"

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
ONNX_PATH = os.path.join(WEIGHTS_DIR, 'resnet34_fruits360.onnx')
DATA_ROOT = os.path.join(
    os.getenv('AXELERA_FRAMEWORK', str(Path.home())), 'data', 'fruits-360-100x100'
)

VAL_SPLIT = 0.15  # 15% for validation
PATIENCE = 2  # For early stopping


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
def clone_repo(repo_url, destination_dir, commit_hash=None):
    # Check if the destination directory already exists
    if os.path.exists(destination_dir):
        print(f"The directory '{destination_dir}' already exists. Skipping cloning.")
        return

    try:
        # Run the git clone command
        subprocess.run(["git", "clone", repo_url, destination_dir], check=True)
        print(f"Repository cloned successfully into '{destination_dir}'")

        # Checkout specific commit if provided
        if commit_hash:
            if commit_hash == DEFAULT_COMMIT_HASH:
                print(f"Using default commit hash: {commit_hash} (141 classes)")
            subprocess.run(["git", "-C", destination_dir, "checkout", commit_hash], check=True)
            print(f"Successfully checked out commit: {commit_hash}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository or checkout commit: {e}")


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
    commit_hash=DEFAULT_COMMIT_HASH,
):
    print(f"{destination_dir = }")
    clone_repo(repo_url, destination_dir, commit_hash)

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
        class_file.parent.mkdir(parents=True, exist_ok=True)
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
    # Ensure model is float32 and on the correct device
    model = model.to(device=DEVICE, dtype=torch.float32)
    since = time.time()
    ensure_cache_dirs()
    best_model_params_path = WEIGHTS

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
                # Move inputs and labels to the target device
                # Move inputs and labels to device with consistent data types
                inputs = to_device(inputs)  # Will convert to float32 if floating point
                labels = to_device(labels, dtype=torch.long)  # Keep labels as integers

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

        # Freeze parameters first if fixed_feature_extractor is True and no weights are loaded
        if fixed_feature_extractor and not (exists_weights and Path(exists_weights).exists()):
            print("Freezing feature extractor parameters.")
            for param in self.model.parameters():
                param.requires_grad = False

        # Modify the final fully connected layer - its parameters will require gradients by default
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(
            num_ftrs, num_classes
        )  # .to(DEVICE) can be removed if model is moved later

        # Move the entire model to the specified device at the end and ensure float32
        # We can't use to_device here because it's for tensors, not models
        self.model = self.model.to(device=DEVICE, dtype=torch.float32)

        # Load existing weights if provided and file exists
        if exists_weights and Path(exists_weights).exists():
            print(f"Loading weights from {exists_weights}")
            # Load weights and ensure they're float32
            state_dict = torch.load(exists_weights, map_location=DEVICE, weights_only=True)
            # Ensure all tensors are float32
            for k in state_dict:
                if (
                    isinstance(state_dict[k], torch.Tensor)
                    and state_dict[k].dtype == torch.float64
                ):
                    state_dict[k] = state_dict[k].to(torch.float32)

            # Handle both direct state dict and model-wrapped state dict
            if any(k.startswith('model.') for k in state_dict.keys()):
                # Remove the 'model.' prefix from the state_dict keys
                new_state_dict = {
                    k[6:]: v for k, v in state_dict.items() if k.startswith('model.')
                }
            else:
                new_state_dict = state_dict

            # Load the state dict with strict=False to allow partial loading
            self.model.load_state_dict(new_state_dict, strict=False)

        # Ensure the final layer is always trainable when using fixed feature extraction
        if fixed_feature_extractor:
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


def transfer_learning(
    train_loader, val_loader, class_names, fixed_feature_extractor=True, num_epochs=6
):
    # If fixed_feature_extractor = True, freeze the weights for all of the network except that
    # of the final fully connected layer. This last fully connected layer is replaced with a
    # new one with random weights and only this layer is trained. If not, we still initialize
    # the model with pretrained weights to finetune the ConvNet.
    # model_conv = build_model(len(class_names), fixed_feature_extractor)
    model_conv = ResNet34Model(
        num_classes=len(class_names), fixed_feature_extractor=fixed_feature_extractor
    )

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
    # Ensure model is float32 and on the correct device
    net = net.to(device=DEVICE, dtype=torch.float32)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader, desc="Measure accuracy on the test dataset"):
            # Move inputs and labels to device with consistent data types
            images = to_device(data[0])  # Will convert to float32 if floating point
            labels = to_device(data[1], dtype=torch.long)  # Keep labels as integers
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
            # Move inputs and labels to device with consistent data types
            images = to_device(data[0])  # Will convert to float32 if floating point
            labels = to_device(data[1], dtype=torch.long)  # Keep labels as integers
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
    # Ensure model is float32 and on the correct device
    model = model.to(device=DEVICE, dtype=torch.float32)
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
            # Move inputs and labels to device with consistent data types
            inputs = to_device(inputs)  # Will convert to float32 if floating point
            labels = to_device(labels, dtype=torch.long)  # Keep labels as integers
            all_samples.extend(list(zip(inputs, labels)))
            if len(all_samples) >= num_images:  # Stop once we have enough samples
                break

        # Randomly sample from collected samples
        import random

        selected_samples = random.sample(all_samples, min(num_images, len(all_samples)))

        for inputs, labels in selected_samples:
            images_so_far += 1
            # Ensure inputs are on the same device as the model
            inputs = to_device(inputs)
            # Unsqueeze to add batch dimension
            batch_inputs = inputs.unsqueeze(0)
            outputs = model(batch_inputs)
            _, preds = torch.max(outputs, 1)

            ax = fig.add_subplot(num_images // 4, 4, images_so_far)
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

    # Create input tensor on the same device as the model
    if input_tensor is None:
        input_tensor = torch.randn(1, 3, 100, 100, device=DEVICE)
    else:
        # Ensure input tensor is on the correct device
        input_tensor = to_device(input_tensor)
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


def main(download_only=False, is_test_mode=False, commit_hash=DEFAULT_COMMIT_HASH):
    try:
        train_loader, val_loader, test_loader, class_names = train_val_test_loaders(
            commit_hash=commit_hash
        )
        if download_only:
            return
        if is_test_mode:
            # Check if weights exist before proceeding with test mode
            if not Path(WEIGHTS).exists():
                print(f"\nERROR: Pre-trained weights not found at {WEIGHTS}")
                print("Please run in training mode first to generate weights:")
                print("  python ax_models/tutorials/resnet34_fruit360.py --train\n")
                sys.exit(1)

            print("Running in test mode with pre-trained weights")
            # Create model with the same number of classes as the dataset and load pre-trained weights
            net = ResNet34Model(
                num_classes=len(class_names), fixed_feature_extractor=True, exists_weights=WEIGHTS
            ).eval()
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

    # Parse commit hash if provided after --download
    commit_hash = DEFAULT_COMMIT_HASH
    if len(sys.argv) > 2 and sys.argv[1] == "--download":
        commit_hash = sys.argv[2]

    # Check if weights file exists to determine default mode
    weights_exist = Path(WEIGHTS).exists()
    default_mode = "test" if weights_exist else "train"

    if "--help" in sys.argv:
        print(f"Usage: {script_name} [--test] [--train] [--download [commit_hash]]")
        print("       --test: Use this flag to run in test mode (using pre-trained weights)")
        print("                This requires pre-trained weights to exist")
        print("       --train: Use this flag to train a new model")
        print(
            f"       Default mode: {'test' if weights_exist else 'train'} (based on whether weights exist)"
        )
        if not weights_exist:
            print(f"       Note: Pre-trained weights not found at {WEIGHTS}")
            print("             Run in training mode first to generate weights")
        print("       --download: Use this flag to only download the dataset and exit")
        print("                 Optional: Provide a commit hash to checkout a specific version")
        print(f"                 Default commit hash: {DEFAULT_COMMIT_HASH} (141 classes)")
        print("                 Example: --download <your-commit-hash>")
    elif "--download" in sys.argv:
        main(download_only=True, commit_hash=commit_hash)
    elif "--train" in sys.argv:
        main(is_test_mode=False, commit_hash=commit_hash)
    elif "--test" in sys.argv:
        main(is_test_mode=True, commit_hash=commit_hash)
    else:
        # Auto-detect mode based on whether weights exist
        if weights_exist:
            print(f"Auto-detected mode: test (pre-trained weights found at {WEIGHTS})")
        else:
            print(f"Auto-detected mode: train (pre-trained weights not found at {WEIGHTS})")
            print("Training a new model...")
        main(is_test_mode=weights_exist, commit_hash=commit_hash)
