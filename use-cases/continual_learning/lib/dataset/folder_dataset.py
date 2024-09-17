# INTEL CONFIDENTIAL
# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the License.

"""Dataset Builder derivatives for TFDS"""
import os
import json
import random
import collections
from logging import getLogger
from typing import Dict, List, Mapping, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import Version
from tensorflow_datasets.core import DatasetInfo
from tensorflow_datasets.core import SplitInfo
from tensorflow_datasets.core import SplitDict
from sklearn.model_selection import train_test_split

from lib.dataset.utils import shard_sequence
from lib.common.utils import progress
from lib.dataset import repository

logger = getLogger(__name__)

_RNG_SEED = 1234

_SUPPORTED_IMAGE_FORMAT = ['.png', '.jpg', '.jpeg']

_TFRECORD_SHARD_TEMPLATE = '{}-{}.tfrecord-{:05d}-of-{:05d}'

_INDEX_FILENAME = 'index.json'

_FEATURES = tfds.features.FeaturesDict({
    'image':
    tfds.features.Image(shape=(None, None, 3)),
    'label':
    tfds.features.ClassLabel(),
    'id': tf.int64,
    'classname': tf.string,
    'filename': tf.string,
})

_Example = collections.namedtuple('_Example', ['image_path', 'label'])

# Dict of 'split_name' -> 'List[_Example]'
SplitExampleDict = Dict[str, List[_Example]]


class ImageDataset(tfds.core.DatasetBuilder):
  """Generic Image Classification Dataset Builder

    `ImageDataset` creates a tf.data.Dataset object, reading images
    directly from the disk.

    This class expects a directory structure as shown:
    ```
    /path/to/root_dir/
        label1/
            xxx.png
            xxy.png
            xxz.png
        label2/
            xxx.png
            xxy.png
            xxz.png
        ...
    ```

    How to use?

    ```
    builder = ImageDataset('/path/to/root_dir',
                                    version='0.1',
                                    index_path='output/index.json',
                                    test_size=0.4)
    print(builder.info) # Magically calculated
    ds = builder.as_dataset(split='train', as_supervised=True)

    # You can export your dataset to a folder with all necessary metadata
    builder.export(out_dir='output/', examples_per_shard=4096)

    # Or directly to a DatasetRepository instance
    builder.export_to_repository(repo)

    # Once exported, use `tfds.builder_from_directory(...)` to create a builder object
    new_builder = tfds.builder_from_directory(builder_dir='output/image_dataset/0.1')
    ds = new_builder.as_dataset(...)
    ```
    """
  VERSION = Version('1.0.0')

  def __init__(
      self,
      root_dir: str,
      *,
      name: str,
      version: str,
      index_path: str,
      test_size: float = 0.4,
  ):
    """
    Constructor for Image Folder dataset builder

    Args:
      root_dir (str):
          Path where images are stored
      name (str):
          Name of the dataset as it would be read by the repository.
      version (str):
          Version string for this dataset stepping (Used to export to repository)
      index_path (str):
          Path to JSON containing class (string) -> label (int) mapping
      test_size (float, optional):
          Test split size from the entire dataset. Defaults to 0.4.
    """
    self.name = name
    super(ImageDataset, self).__init__()
    self._version = version
    self._data_dir = root_dir

    # Extract examples, labels
    root_dir = os.path.expanduser(root_dir)
    self._examples, unique_labels = _get_all_label_images(root_dir)

    # Rebuild/Create index
    self._index = self.build_index(labels=unique_labels, index_path=index_path)
    self.info.features['label'].names = list(self._index.keys())

    # Generate train/test splits
    logger.info(f'Creating test split with ratio: {test_size}')
    self._split_examples = _stratified_train_test_split(self._examples,
                                                        test_size=test_size)

    # Update DatasetInfo splits
    split_infos = [
        SplitInfo(
            name=split_name,
            num_bytes=0,
            shard_lengths=[len(examples)],
        ) for split_name, examples in self._split_examples.items()
    ]
    split_dict = SplitDict(split_infos)
    self.info.set_splits(split_dict)

  def build_index(self, labels: List[str],
                  index_path: Optional[str]) -> Mapping[str, int]:
    """Builds an index of labels, assiging a unique integer to each class.

        If `index_path` is provided, overlapping classes will be assigned
        label that is set in the JSON at `index_path`.

        Args:
            index_path (str, optional): Path to index.json to be used for update.
        """
    # Load index
    index = dict()
    if tf.io.gfile.exists(index_path):
      with tf.io.gfile.GFile(index_path, 'r') as f:
        index = json.load(f)
      logger.info(f'Read {index_path} ({len(index)} existing entries)')
    else:
      logger.info(
          f'Path {index_path} does not exist. New index will be generated.')

    # Update index
    new_label = 0 if bool(index) is False else len(index)
    newly_added = []
    for name in labels:
      if name not in index:
        index[name] = new_label
        new_label += 1
        newly_added.append(name)
    logger.info(f'New entries: {newly_added}')

    return index

  def export_to_repository(self,
                           repo: repository.DatasetRepository,
                           examples_per_shard: int = -1):
    """Export current dataset to a repository as a TFRecord"""
    if self._version in repo.versions:
      raise Exception(
          f'Provided version {self._version} already exists in {repo.location}\n'
          'Assign another version to continue.')

    self.export(out_dir=repo.location, examples_per_shard=examples_per_shard)

    # Write index to disk
    index_path = tf.io.gfile.join(repo.location, _INDEX_FILENAME)
    with open(index_path, 'w') as f:
      json.dump(self._index, f, indent=0, ensure_ascii=False)
    logger.info(f'Class Index updated at {index_path}')

  def export(self, out_dir: str, examples_per_shard: int = -1):
    """Exports dataset to one (or more) TFRecords with metadata.

    Args:
      out_dir (str):
        Directory where TFRecords are to be written.
      examples_per_shard (int, optional):
        Examples to serialize per shard.
        -1 implies all examples are written to single record.
        Defaults to -1.
    """
    logger.info(
        f'Examples per shard: {examples_per_shard} (-1 == single record)')

    for split_name in self._split_examples:
      examples = self._split_examples[split_name]

      # Create parent dir if not exist
      record_dir = os.path.join(out_dir, self._version)
      os.makedirs(record_dir, exist_ok=True)

      # Create shards and write each
      example_shards = shard_sequence(examples, n=examples_per_shard)
      for i, shard in enumerate(example_shards):
        record_name = _TFRECORD_SHARD_TEMPLATE.format(self.info.name,
                                                      split_name, i,
                                                      len(example_shards))
        out_path = os.path.join(record_dir, record_name)
        self._write_tfrecord(examples=shard, out_path=out_path)

      logger.info(f'Done writing {split_name} split at: {record_dir}')
      logger.info(
          f'Number of examples: {len(examples)} (shards={len(example_shards)})'
      )

    # Compute splitInfo for TFRecord folder
    self._write_metadata(record_dir)

  def _write_metadata(self, data_dir: str) -> None:
    # Compute splitInfo from TFRecords
    split_infos = tfds.folder_dataset.compute_split_info_from_directory(
        data_dir=data_dir, out_dir=data_dir)

    # Write TFDS metadata for easy-access via builder
    tfds.folder_dataset.write_metadata(
        data_dir=data_dir,
        features=self.info.features,
        split_infos=split_infos,
        supervised_keys=self.info.supervised_keys)
    logger.info(f'Dataset exported with metadata: {data_dir}')

  def _write_tfrecord(self, examples: List[_Example], out_path: str) -> None:
    with tf.io.TFRecordWriter(out_path) as writer:
      for id, example in enumerate(progress(examples, desc=out_path)):

        with tf.io.gfile.GFile(example.image_path, 'rb') as f:
          img_bytes = f.read()

        ex_bytes = self.info.features.serialize_example({
            'image':
            img_bytes,
            'label':
            self.info.features['label'].str2int(example.label),
            'id': id,
            'filename': example.image_path,
            'classname': example.label,
        })
        writer.write(ex_bytes)

  def _download_and_prepare(self, **kwargs):
    raise NotImplementedError(
        f'Need not call download_and_prepare function for {type(self).__name__}.'
    )

  def _info(self) -> DatasetInfo:
    return DatasetInfo(
        builder=self,
        description='Generic Image Classification Dataset',
        features=_FEATURES,
        supervised_keys=('image', 'label'),
        homepage='Undefined',
    )

  def _as_dataset(
      self,
      split: str,
      shuffle_files: bool = True,
      decoders: Optional[Dict[str, tfds.decode.Decoder]] = None,
      read_config: tfds.ReadConfig = None,
  ) -> tf.data.Dataset:
    """Generate dataset for the given split"""
    del read_config  # Maybe?

    if split not in self.info.splits.keys():
      raise ValueError('Unrecognized split {}. '
                       'Split name should be one of {}.'.format(
                           split, list(self.info.splits.keys())))

    # Extract image paths and labels
    image_paths = list()
    labels = list()
    examples = self._split_examples[split]
    for example in examples:
      image_paths.append(example.image_path)
      labels.append(self.info.features['label'].str2int(example.label))

    # Build tf.data.Dataset object
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle_files:
      ds.shuffle(len(examples))

    # Fuse load/decode
    def _load_and_decode_fn(*args, **kwargs):
      ex = _load_example(*args, **kwargs)
      return self.info.features.decode_example(ex, decoders=decoders)

    ds = ds.map(_load_and_decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def _load_example(path: tf.Tensor, label: tf.Tensor) -> Dict[str, tf.Tensor]:
  img = tf.io.read_file(path)
  return {
      'image': img,
      'label': tf.cast(label, tf.int64),
  }


def _get_all_label_images(root_dir: str) -> Tuple[List[_Example], List[str]]:
  """Extract all label names and associated images

  Args:
      root_dir (str): Directory where root_dir/label/image.png are located

  Returns:
      examples: List of namedtuple(image_path, label)
      labels: Set of labels found
  """
  examples = []
  labels = set()

  for label_name in sorted(_list_folders(root_dir)):
    labels.add(label_name)
    examples.extend([
        _Example(image_path=image_path, label=label_name)
        for image_path in sorted(
            _list_img_paths(os.path.join(root_dir, label_name)))
    ])

  # Shuffle the images deterministically
  rng = random.Random(_RNG_SEED)
  rng.shuffle(examples)
  return examples, sorted(labels)


def _list_folders(root_dir: str) -> List[str]:
  return [
      f for f in tf.io.gfile.listdir(root_dir)
      if tf.io.gfile.isdir(os.path.join(root_dir, f))
  ]


def _list_img_paths(root_dir: str) -> List[str]:
  return [
      os.path.join(root_dir, f) for f in tf.io.gfile.listdir(root_dir)
      if any(f.lower().endswith(ext) for ext in _SUPPORTED_IMAGE_FORMAT)
  ]


def _stratified_train_test_split(examples: List[_Example],
                                 test_size: float) -> SplitExampleDict:
  image_paths, labels = zip(*examples)
  X_train, X_test, y_train, y_test = train_test_split(image_paths,
                                                      labels,
                                                      test_size=test_size,
                                                      shuffle=True,
                                                      random_state=_RNG_SEED,
                                                      stratify=labels)
  split_examples = collections.defaultdict(list)
  split_examples['train'] = list(
      map(lambda x, y: _Example(x, y), X_train, y_train))
  split_examples['test'] = list(
      map(lambda x, y: _Example(x, y), X_test, y_test))
  return split_examples
