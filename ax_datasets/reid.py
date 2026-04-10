# Copyright Axelera AI, 2025
from __future__ import annotations

import copy
import glob
import os.path as osp
import random
import re

from PIL import Image
import torch

from axelera import types
from axelera.app import data_utils, eval_interfaces, logging_utils

LOG = logging_utils.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Base class for ReID datasets with support for balanced sampling.

    This class provides functionality for loading and organizing person re-identification
    datasets with proper interleaving of query and gallery samples to ensure balanced
    evaluation when processing a limited number of frames.

    Args:
        train (list): Tuples of (img_path(s), pid, camid).
        query (list): Tuples of (img_path(s), pid, camid).
        gallery (list): Tuples of (img_path(s), pid, camid).
        transform: Transform function for image preprocessing.
        k_tfm (int): Number of times to apply augmentation to an image independently.
        mode (str): 'train', 'query', 'gallery', or 'query_and_gallery'.
        combineall (bool): Combines train, query and gallery in a dataset for training.
        verbose (bool): Show dataset information.
    """

    # junk_pids contains useless person IDs, e.g. background,
    # false detections, distractors. These IDs will be ignored
    # when combining all images in a dataset for training, i.e.
    # combineall=True
    _junk_pids = []

    # Some datasets are only used for training, like CUHK-SYSU
    # In this case, "combineall=True" is not used for them
    _train_only = False

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode='train',
        combineall=False,
        verbose=True,
        **kwargs,
    ):
        # Standardize data format
        train, query, gallery = self._standardize_data_format(train, query, gallery)

        if len(query) > 0 and len(gallery) > 0:
            query, gallery = self._organize_data_by_identity(query, gallery, verbose)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.query_and_gallery = self._create_interleaved_dataset(query, gallery, verbose)

        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)

        if self.combineall:
            self.combine_all()

        # Set the active dataset based on mode
        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        elif self.mode == 'query_and_gallery':
            self.data = self.query_and_gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery | query_and_gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def _standardize_data_format(self, train, query, gallery):
        """Standardize data format to ensure consistent tuples with split_name and dsetid.

        Args:
            train, query, gallery: Lists of data tuples

        Returns:
            Standardized train, query, gallery lists
        """
        # First, ensure we have split_name
        if len(train) > 0 and len(train[0]) == 3:
            train = [(img_path, pid, camid, "train") for img_path, pid, camid in train]
        if len(query) > 0 and len(query[0]) == 3:
            query = [(img_path, pid, camid, "query") for img_path, pid, camid in query]
        if len(gallery) > 0 and len(gallery[0]) == 3:
            gallery = [(img_path, pid, camid, "gallery") for img_path, pid, camid in gallery]

        # Then ensure we have dsetid
        if len(train) > 0 and len(train[0]) == 4:
            train = [(*items, 0) for items in train]
        if len(query) > 0 and len(query[0]) == 4:
            query = [(*items, 0) for items in query]
        if len(gallery) > 0 and len(gallery[0]) == 4:
            gallery = [(*items, 0) for items in gallery]

        return train, query, gallery

    def _organize_data_by_identity(self, query, gallery, verbose=True):
        """Organize query and gallery data by person identity for better distribution.

        This function ensures that when processing a limited number of frames:
        1. Common person IDs between query and gallery are prioritized first
        2. Person IDs are shuffled to avoid processing batches of the same person
        3. Samples within each person ID are also shuffled for better distribution

        The goal is to maximize the chance of having corresponding identities
        in query and gallery sets even when using a subset of the data.

        Args:
            query: List of query data tuples
            gallery: List of gallery data tuples
            verbose: Whether to print information

        Returns:
            Reorganized query and gallery lists with balanced person identity distribution
        """
        # Find common person IDs between query and gallery sets
        query_pids = [item[1] for item in query]
        gallery_pids = [item[1] for item in gallery]

        query_pid_set = set(query_pids)
        gallery_pid_set = set(gallery_pids)
        common_pids = query_pid_set.intersection(gallery_pid_set)

        LOG.trace(f"Found {len(common_pids)} common person IDs between query and gallery")
        LOG.trace(f"Query: {len(query_pids)} images, {len(query_pid_set)} unique IDs")
        LOG.trace(f"Gallery: {len(gallery_pids)} images, {len(gallery_pid_set)} unique IDs")

        # Organize items by person ID
        pid_to_query_items = {}
        for item in query:
            pid = item[1]
            if pid not in pid_to_query_items:
                pid_to_query_items[pid] = []
            pid_to_query_items[pid].append(item)

        pid_to_gallery_items = {}
        for item in gallery:
            pid = item[1]
            if pid not in pid_to_gallery_items:
                pid_to_gallery_items[pid] = []
            pid_to_gallery_items[pid].append(item)

        # Create ordered lists prioritizing common IDs
        random.seed(42)  # For reproducibility

        # Prepare ID lists
        common_pid_list = list(common_pids)
        query_only_pids = list(query_pid_set - common_pids)
        gallery_only_pids = list(gallery_pid_set - common_pids)

        # Shuffle to avoid any bias
        random.shuffle(common_pid_list)
        random.shuffle(query_only_pids)
        random.shuffle(gallery_only_pids)

        # Reorder query and gallery lists
        query_reordered = []
        gallery_reordered = []

        # Add items with common IDs first
        for pid in common_pid_list:
            query_items = pid_to_query_items.get(pid, [])
            gallery_items = pid_to_gallery_items.get(pid, [])
            random.shuffle(query_items)
            random.shuffle(gallery_items)
            query_reordered.extend(query_items)
            gallery_reordered.extend(gallery_items)

        # Then add items without common IDs
        for pid in query_only_pids:
            query_items = pid_to_query_items.get(pid, [])
            random.shuffle(query_items)
            query_reordered.extend(query_items)

        for pid in gallery_only_pids:
            gallery_items = pid_to_gallery_items.get(pid, [])
            random.shuffle(gallery_items)
            gallery_reordered.extend(gallery_items)

        return query_reordered, gallery_reordered

    def _create_interleaved_dataset(self, query, gallery, verbose=True):
        """Create a dataset that interleaves query and gallery samples in batches.

        This function builds on the identity organization from _organize_data_by_identity
        to create a single dataset that alternates between query and gallery samples
        in proportion to their original sizes. This ensures:

        1. Balanced representation of both query and gallery sets
        2. When using limited samples, both sets are fairly represented
        3. The identity distribution from _organize_data_by_identity is preserved

        Args:
            query: List of query data tuples, already organized by person identity
            gallery: List of gallery data tuples, already organized by person identity
            verbose: Whether to print information

        Returns:
            A single interleaved list combining query and gallery samples
        """
        if not query or not gallery:
            return query + gallery

        # Interleave query and gallery based on ratio to ensure balanced sampling
        query_and_gallery_interleaved = []

        # Calculate ratio between query and gallery for proper interleaving
        total_items = len(query) + len(gallery)
        query_ratio = len(query) / total_items
        gallery_ratio = len(gallery) / total_items

        # Determine batch sizes for interleaving (ensure at least 1)
        query_batch = max(1, int(10 * query_ratio))
        gallery_batch = max(1, int(10 * gallery_ratio))

        # Create interleaved list with batch sampling
        query_idx = 0
        gallery_idx = 0

        while query_idx < len(query) or gallery_idx < len(gallery):
            # Add batch of query items
            for _ in range(query_batch):
                if query_idx < len(query):
                    query_and_gallery_interleaved.append(query[query_idx])
                    query_idx += 1

            # Add batch of gallery items
            for _ in range(gallery_batch):
                if gallery_idx < len(gallery):
                    query_and_gallery_interleaved.append(gallery[gallery_idx])
                    gallery_idx += 1

        if verbose:
            # Verify interleaving balance
            first_100_items = query_and_gallery_interleaved[
                : min(100, len(query_and_gallery_interleaved))
            ]
            query_count = sum(1 for item in first_100_items if item[3] == "query")
            gallery_count = sum(1 for item in first_100_items if item[3] == "gallery")
            LOG.info(
                f"Interleaved dataset first 100 items: {query_count} query, {gallery_count} gallery"
            )

        return query_and_gallery_interleaved

    def __getitem__(self, index):
        item_data = self.data[index]
        img_path = item_data[0]
        pid = item_data[1]
        camid = item_data[2]
        split_name = item_data[3] if len(item_data) > 3 else ""
        dsetid = item_data[4] if len(item_data) > 4 else 0

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid,
            'split_name': split_name,
        }
        return item

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid, camid, dsetid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            dsetid += self.num_datasets
            train.append((img_path, pid, camid, dsetid))

        ###################################
        # Note that
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset; setting it to True will
        #    create new IDs that should have already been included
        ###################################
        if isinstance(train[0][0], str):
            return ImageDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
            )
        else:
            raise ValueError(
                'Invalid dataset. Got {}, but expected to be '
                'one of [ImageDataset]'.format(type(other))
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_num_pids(self, data):
        """Returns the number of training person identities.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set()
        for items in data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self, data):
        """Returns the number of training cameras.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        cams = set()
        for items in data:
            camid = items[2]
            cams.add(camid)
        return len(cams)

    def get_num_datasets(self, data):
        """Returns the number of datasets included.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[3]
            dsets.add(dsetid)
        return len(dsets)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        if self._train_only:
            return

        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for items in self.gallery:
            pid = items[1]
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for item in data:
                pid = item[1]
                if pid in self._junk_pids:
                    continue
                # Create a new tuple with the relabeled pid
                new_item = list(item)
                new_item[1] = pid2label[pid] + self.num_train_pids
                combined.append(tuple(new_item))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        msg = (
            '  ----------------------------------------\n'
            '  subset   | # ids | # items | # cameras\n'
            '  ----------------------------------------\n'
            '  train    | {:5d} | {:7d} | {:9d}\n'
            '  query    | {:5d} | {:7d} | {:9d}\n'
            '  gallery  | {:5d} | {:7d} | {:9d}\n'
            '  ----------------------------------------\n'
            '  items: images/tracklets for image/video dataset\n'.format(
                num_train_pids,
                len(self.train),
                num_train_cams,
                num_query_pids,
                len(self.query),
                num_query_cams,
                num_gallery_pids,
                len(self.gallery),
                num_gallery_cams,
            )
        )

        return msg

    def _transform_image(self, tfm, k_tfm, img0):
        """Transforms a raw image (img0) k_tfm times with
        the transform function tfm.
        """
        img_list = []

        for k in range(k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        item_data = self.data[index]
        img_path = item_data[0]
        pid = item_data[1]
        camid = item_data[2]
        split_name = item_data[3] if len(item_data) > 3 else ""
        dsetid = item_data[4] if len(item_data) > 4 else 0

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid,
            'split_name': split_name,
        }
        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        LOG.info('=> Loaded {}'.format(self.__class__.__name__))
        LOG.info('  ----------------------------------------')
        LOG.info('  subset   | # ids | # images | # cameras')
        LOG.info('  ----------------------------------------')
        LOG.info(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        LOG.info(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        LOG.info(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        LOG.info('  ----------------------------------------')


class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """

    _junk_pids = [0, -1]
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.dataset_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True, split_name="train")
        query = self.process_dir(self.query_dir, relabel=False, split_name="query")
        gallery = self.process_dir(self.gallery_dir, relabel=False, split_name="gallery")
        if self.market1501_500k:
            gallery += self.process_dir(
                self.extra_gallery_dir, relabel=False, split_name="gallery"
            )

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False, split_name=""):
        """Processes a directory and adds a split flag to each item."""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, split_name))  # Add split flag

        return data


# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    },
}


class MSMT17(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """

    dataset_dir = 'msmt17'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'MSMT17 ReID dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(self.dataset_dir, main_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, main_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, main_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'list_gallery.txt')

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path, split_name="train")
        val = self.process_dir(self.train_dir, self.list_val_path, split_name="val")
        query = self.process_dir(self.test_dir, self.list_query_path, split_name="query")
        gallery = self.process_dir(self.test_dir, self.list_gallery_path, split_name="gallery")

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if 'combineall' in kwargs and kwargs['combineall']:
            train += val

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path, split_name=""):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for _, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid, split_name))

        return data


class DanceTrackReId(ImageDataset):
    dataset_dir = 'dancetrack-reid'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True, split_name="train")
        query = self.process_dir(self.query_dir, relabel=False, split_name="query")
        gallery = self.process_dir(self.gallery_dir, relabel=False, split_name="gallery")

        super(DanceTrackReId, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False, split_name=""):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            base_name = osp.basename(img_path)
            parts = base_name.split('_')
            if len(parts) < 2:
                continue

            pid = int(parts[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            base_name = osp.basename(img_path)
            parts = base_name.split('_')
            if len(parts) < 2:
                continue
            pid = int(parts[0])

            # DanceTrack is effectively single-camera. ReID eval (Market1501-style) drops
            # same-pid + same-cam matches, which would wipe out all positives. Force
            # query/gallery into different camids so valid matches remain.
            camid = 0
            if split_name == "gallery":
                # Must differ from query camid for the same reason above.
                camid = 1

            if relabel:
                pid = pid2label[pid]

            data.append((img_path, pid, camid, split_name))

        return data


class DanceTrackReIdDataAdapter(types.DataAdapter):
    def __init__(self, dataset_config, model_info):
        self.dump_embeddings_to = dataset_config.get('dump_embeddings_to', '')
        self.dataset_config = dataset_config
        self.model_info = model_info

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([item['img'] for item in batched_data], 0)
        )

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='DanceTrack_ReId',
            data_root_dir=root,
            split='val',
            is_private=True,
        )

        return torch.utils.data.DataLoader(
            DanceTrackReId(root, mode='train', transform=transform),
            batch_size=batch_size,
            shuffle=True,
            generator=kwargs.get('generator'),
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='DanceTrack_ReId',
            data_root_dir=root,
            split='val',
            is_private=True,
        )

        return torch.utils.data.DataLoader(
            DanceTrackReId(root, mode='query_and_gallery'),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_validation(self, batched_data):
        return [
            types.FrameInput.from_image(
                img=item['img'],
                color_format=types.ColorFormat.RGB,
                ground_truth=eval_interfaces.ReIdGtSample(
                    person_id=item['pid'], camera_id=item['camid'], split_name=item['split_name']
                ),
                img_id='',
            )
            for item in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators.reid import ReIdEvaluator

        return ReIdEvaluator(dump_embeddings_to=self.dump_embeddings_to)


class MSMT17DataAdapter(types.DataAdapter):
    def __init__(self, dataset_config, model_info):
        self.dump_embeddings_to = dataset_config.get('dump_embeddings_to', '')
        self.dataset_config = dataset_config
        self.model_info = model_info

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([item['img'] for item in batched_data], 0)
        )

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='MSMT17ReIdDataset',
            data_root_dir=root,
            split='val',
            is_private=True,
        )

        return torch.utils.data.DataLoader(
            MSMT17(root, mode='train', transform=transform),
            batch_size=batch_size,
            shuffle=True,
            generator=kwargs.get('generator'),
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='MSMT17ReIdDataset',
            data_root_dir=root,
            split='val',
            is_private=True,
        )

        return torch.utils.data.DataLoader(
            MSMT17(root, mode='query_and_gallery'),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_validation(self, batched_data):
        return [
            types.FrameInput.from_image(
                img=item['img'],
                color_format=types.ColorFormat.RGB,
                ground_truth=eval_interfaces.ReIdGtSample(
                    person_id=item['pid'], camera_id=item['camid'], split_name=item['split_name']
                ),
                img_id='',
            )
            for item in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators.reid import ReIdEvaluator

        return ReIdEvaluator(dump_embeddings_to=self.dump_embeddings_to)


class Market1501DataAdapter(types.DataAdapter):
    def __init__(self, dataset_config, model_info):
        self.dump_embeddings_to = dataset_config.get('dump_embeddings_to', '')
        self.dataset_config = dataset_config
        self.model_info = model_info

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([item['img'] for item in batched_data], 0)
        )

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='Market1501ReIdDataset',
            data_root_dir=root,
            split='val',
            is_private=False,
        )

        return torch.utils.data.DataLoader(
            Market1501(root, mode='train', transform=transform),
            batch_size=batch_size,
            shuffle=True,
            generator=kwargs.get('generator'),
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='Market1501ReIdDataset',
            data_root_dir=root,
            split='val',
            is_private=False,
        )

        return torch.utils.data.DataLoader(
            Market1501(root, mode='query_and_gallery'),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_validation(self, batched_data):
        return [
            types.FrameInput.from_image(
                img=item['img'],
                color_format=types.ColorFormat.RGB,
                ground_truth=eval_interfaces.ReIdGtSample(
                    person_id=item['pid'], camera_id=item['camid'], split_name=item['split_name']
                ),
                img_id='',
            )
            for item in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):

        from ax_evaluators.reid import ReIdEvaluator

        return ReIdEvaluator(dump_embeddings_to=self.dump_embeddings_to)
