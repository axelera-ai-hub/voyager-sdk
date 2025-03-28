# Copyright Axelera AI, 2024
# DataLoader for TensorFlow
from typing import Any

import PIL
import tensorflow as tf
import torch
from torch.utils import data

from axelera import types
from axelera.app import eval_interfaces
from axelera.app.yaml import MapYAMLtoFunction


class TFRecordDataset(data.Dataset):
    '''Torch Dataset from TFRecord files for classification.
    To extend to the other tasks, override the _parse_tfrecord method.'''

    def __init__(self, transform, root, split, args):
        supported_args = ['binary_img']
        defaults = {'binary_img': False}

        if split == 'cal':
            supported_args.append('cal_data')
        elif split == 'val':
            supported_args.append('val_data')
        else:
            raise ValueError(f"Invalid split: {split}")

        yargs = MapYAMLtoFunction(
            supported=supported_args,
            required=[],
            defaults=defaults,
            named_args=['split'],
            attribs=args,
        )

        self.tfrecord_path = root / yargs.get_arg(supported_args[-1])
        self.transform = transform
        self.binary_img = yargs.get_arg('binary_img')
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.tfrecord_path.is_dir():
            # If it's a directory, iterate over all TFRecord files
            file_paths = list(self.tfrecord_path.glob('*.tfrecord'))
            return [record for path in file_paths for record in tf.data.TFRecordDataset(str(path))]
        elif self.tfrecord_path.is_file():
            # If it's a file, load it directly
            return list(tf.data.TFRecordDataset(str(self.tfrecord_path)))
        else:
            raise ValueError(f"{self.tfrecord_path} is neither a valid file nor a directory")

    def __len__(self):
        return len(self.dataset)

    def _parse_tfrecord(self, tfrecord):
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64)}

        if self.binary_img:
            features['image/encoded'] = tf.io.FixedLenFeature([], tf.string)
            features['image/filename'] = tf.io.FixedLenFeature([], tf.string)
        else:
            features['image/img_path'] = tf.io.FixedLenFeature([], tf.string)

        data = tf.io.parse_single_example(tfrecord, features)

        if self.binary_img:
            tf_image = tf.image.decode_jpeg(data['image/encoded'], channels=3)
            # here already assume the image is in RGB format
            image = PIL.Image.fromarray(tf_image.numpy())
        else:
            image = PIL.Image.open(data['image/img_path'].numpy())

        if self.transform:
            image = self.transform(image)

        label = data['image/source_id'].numpy()
        if self.binary_img:
            filename = data['image/filename'].numpy().decode('utf-8')
            return image, label, filename
        else:
            return image, label

    def __getitem__(self, idx):
        tfrecord = self.dataset[idx]
        return self._parse_tfrecord(tfrecord)


class TfrecordsClassificationDataAdapter(types.DataAdapter):
    """Data adapter for classification task using tfrecord.

    examples:
        FaceDataset:
            class: TfrecordsClassificationDataAdapter
            data_dir_name: LFW/lfw
            labels_path: lfw.names
            repr_imgs: Aaron_Peirsol
            # cal_data: cal.tfrecord
            val_data: lfw.tfrecord
            binary_img: False
    """

    def __init__(self, dataset_config, model_info):
        if not dataset_config:
            raise Exception("Model requires a dataset to calibrate")

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        if repr_dataloader := self.check_representative_images(transform, batch_size, **kwargs):
            return repr_dataloader
        return data.DataLoader(
            TFRecordDataset(transform, root, 'cal', **kwargs),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        return data.DataLoader(
            TFRecordDataset(None, root, 'val', **kwargs),
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

    def reformat_for_validation(self, batched_data: Any):
        return self._format_measurement_data(batched_data)

    def _format_measurement_data(self, batched_data: Any) -> list[types.FrameInput]:
        return [
            types.FrameInput(
                img=types.Image.fromany(image),
                ground_truth=eval_interfaces.ClassificationGroundTruthSample(class_id=target),
                img_id=optional[0] if optional else '',
            )
            for image, target, *optional in batched_data
        ]

    # TODO: implement after we refactor fmzoo.eval.evaluator.EvaluatorFull as a ax_evaluator
    # def evaluator(self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False):
