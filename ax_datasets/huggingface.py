# Copyright Axelera AI, 2024
import re
from typing import Any, Tuple

from datasets import load_dataset
import numpy as np
import torch

from axelera import types
from axelera.app import eval_interfaces
from axelera.app.model_utils import box
from axelera.app.yaml import MapYAMLtoFunction


class HuggingFaceDataset:
    def __init__(self, dataset_name, data_dir, transform, split, args):
        self.dataset_name = dataset_name
        self.transform = transform

        # Map YAML arguments to Hugging Face dataset parameters
        yargs = MapYAMLtoFunction(
            supported=['other_params'],
            required=[],
            defaults={},
            named_args=[],
            attribs=args,
        )

        other_params = yargs.get_kwargs()
        self.dataset = load_dataset(
            self.dataset_name,
            split=split,
            data_dir=data_dir,
            trust_remote_code=True,
            **other_params,
        )
        # TODO: see if it is possible to use the built-in with_transform or set_transform
        # self.dataset.set_transform(self.transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        item = self.dataset[index]
        if self.transform is not None and 'image' in item:
            item['image'] = self.transform(item['image'])
        return item

    def __len__(self):
        return len(self.dataset)


class HuggingFaceObjDetDataAdapter(types.DataAdapter):
    """Data adapter for accessing HuggingFace datasets.

    Supported datasets:
    - wider_face

    examples:
        HuggingFace-WiderFace:
            class: HuggingFaceObjDetDataAdapter
            data_dir_name: wider_face # if no specified, follow HuggingFace convention
            dataset_name: wider_face
            labels_path: $AXELERA_FRAMEWORK/ax_datasets/labels/face.names
            repr_imgs_dir_path: $AXELERA_FRAMEWORK/data/coco2017_400_b680128
            repr_imgs_url: https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/coco2017_repr400.zip
            repr_imgs_md5: b680128512392586e3c86b670886d9fa
    """

    SUPPORTED_DATASETS = re.findall(r'^\s*-\s*(\w+)\s*$', __doc__, re.M)

    def __init__(self, dataset_config, model_info):
        self.dataset_name = dataset_config.get('dataset_name')
        if not self.dataset_name:
            raise ValueError("Please specify dataset_name in YAML")

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {self.dataset_name} is not supported. Supported datasets: {self.SUPPORTED_DATASETS}"
            )

        # HuggingFace has its own cache directory; here we follow their convention
        # instead of using the root directory if data_dir_name is not specified or empty
        self.data_dir = dataset_config.get('data_dir_name') or None

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        if repr_dataloader := self.check_representative_images(transform, batch_size, **kwargs):
            return repr_dataloader
        return torch.utils.data.DataLoader(
            HuggingFaceDataset(self.dataset_name, self.data_dir, transform, 'train', kwargs),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        return torch.utils.data.DataLoader(
            # Hugging Face uses 'validation' instead of 'val'
            HuggingFaceDataset(self.dataset_name, self.data_dir, None, 'validation', kwargs),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([data['image'] for data in batched_data], 0)
        )

    def reformat_for_validation(self, batched_data: Any):
        return self._format_measurement_data(batched_data)

    def _format_measurement_data(self, batched_data: Any) -> list[types.FrameInput]:
        batched_unified_data = []
        for data in batched_data:
            if self.dataset_name == "wider_face":
                # https://huggingface.co/datasets/wider_face
                xyxy = box.ltwh2xyxy(np.array(data['faces']['bbox']))
            else:
                raise NotImplementedError(f"Dataset {self.dataset_name} is not supported for now")
            batched_unified_data.append(
                types.FrameInput.from_image(
                    img=data['image'],
                    # huggingface datasets use PIL defaultly
                    color_format=types.ColorFormat.RGB,
                    ground_truth=eval_interfaces.ObjDetGroundTruthSample.from_list(
                        boxes=xyxy, labels=data['category_id'], img_id=data['image_id']
                    ),
                    img_id='',
                )
            )
        return batched_unified_data

    # def evaluator(self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False):
