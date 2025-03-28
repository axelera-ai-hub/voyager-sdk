# Copyright Axelera AI, 2024
from typing import Any

import numpy as np
import torch
import torch.utils.data as data

from axelera import types
from axelera.app import data_utils, eval_interfaces, logging_utils

LOG = logging_utils.getLogger(__name__)

# The dataset implementation is taken from
# https://github.com/biubug6/Pytorch_Retinaface/blob/master/data/wider_face.py
# which is licensed under MIT License. The only change is to return numpy.ndarray
# instead of torch.Tensor from __getitem__
class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        import cv2

        self.imread = cv2.imread

        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = self.imread(self.imgs_path[index])
        height, width = data_utils.get_image_size(img)

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            if len(label) > 4:  # Check if landmarks are available
                annotation[0, 4] = label[4]  # l0_x
                annotation[0, 5] = label[5]  # l0_y
                annotation[0, 6] = label[7]  # l1_x
                annotation[0, 7] = label[8]  # l1_y
                annotation[0, 8] = label[10]  # l2_x
                annotation[0, 9] = label[11]  # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if annotation[0, 4] < 0:
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1
            else:
                annotation[0, 4:15] = -1  # Set landmarks to -1 if not available

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return img, target, self.imgs_path[index]


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim

    Note:
        From https://github.com/biubug6/Pytorch_Retinaface/blob/master/data/wider_face.py
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


class WiderFaceDataAdapter(types.DataAdapter):
    """Data adapter for accessing WiderFace datasets for face detection and landmark localization"""

    def __init__(self, dataset_config, model_info):
        pass

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        if repr_dataloader := self.check_representative_images(transform, batch_size, **kwargs):
            return repr_dataloader

        data_utils.check_and_download_dataset(
            dataset_name='WiderFace', data_root_dir=root, split='train'
        )
        txt_path = root / kwargs.get('cal_data', 'train/label.txt')
        return torch.utils.data.DataLoader(
            WiderFaceDetection(str(txt_path)),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=detection_collate,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='WiderFace', data_root_dir=root, split='val'
        )
        txt_path = root / kwargs.get('val_data', 'val/label.txt')
        return torch.utils.data.DataLoader(
            WiderFaceDetection(str(txt_path)),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([data[0] for data in batched_data], 0)
        )

    def reformat_for_validation(self, batched_data: Any) -> list[types.FrameInput]:
        return [
            types.FrameInput.from_image(
                img=img,
                color_format=types.ColorFormat.BGR,
                ground_truth=eval_interfaces.GeneralSample(annotations=target),
                img_id=img_path,
            )
            for img, target, img_path in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators.widerface import WiderFaceEvaluator

        return WiderFaceEvaluator(dataset_root=dataset_root)
