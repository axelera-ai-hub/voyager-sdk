# Axelera class for PyTorch tutorial
# Copyright Axelera AI, 2024

from pathlib import Path

import torch
from tutorial_cifar10 import WEIGHTS, TutorialCIFAR10, test, testloader, train, trainloader

from axelera import types


class TutorialCIFAR10(TutorialCIFAR10, types.Model):
    def __init__(self, **kwargs):
        super().__init__()

    def init_model_deploy(self):
        if not Path(WEIGHTS).exists():
            print(f"First train model to generate weights ('{WEIGHTS}')")
            train(self, trainloader())
            test(self, testloader())
        self.load_state_dict(torch.load(WEIGHTS))


class TutorialCIFAR10Adaptor(types.DataAdapter):
    def create_calibration_data_loader(self, transform, batch, root, **kwargs):
        return trainloader(transform, batch_size=batch, root=root)

    def create_inference_dataloader(self, root, **kwargs):
        # Configure datalaoder to produce raw images
        # (transformed on inference device)
        return testloader(batch_size=1, transform=None, root=root)
