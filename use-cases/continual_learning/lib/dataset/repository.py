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

from logging import getLogger
from typing import List, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

logger = getLogger(__name__)

_DATASET_CONFIG = 'dataset_info.json'
_INDEX_FILENAME = 'index.json'


class DatasetRepository:
  """Dataset Version Control API.
    Use this class to manage versions of datasets created by
    `tfds.core.DatasetBuilder` (or `ImageFolder.export(...)`)

    Usage:
    ```
    # Creating a new repository for a dataset
    repository = DatasetRepository(data_dir='/tmp/repository',
                                   dataset_name='vege')
    repository.versions  # []

    # Access attributes
    repository.location  # /tmp/repository/vege
    repository.name      # vege

    # Access an existing repository
    repository = DatasetRepository(data_dir='/tmp/repository',
                                   dataset_name='vege')
    repository.versions  # ['1.1', '1.2']

    # Return a dataset builder for specific versions
    subset_builder = repository.get_builder(versions=['1.1', '1.2'])
    ds = subset_builder.as_dataset(...)

    # Create a dataset from all available versions
    builder_all = repository.get_builder(repository.versions) # Or pass nothing
    ds = builder_all.as_dataset(...)

    # Delete a version from the repository (only one version at a time)
    repository.purge(version='1.1')
    ```
    """

  def __init__(self, *, data_dir: str, initialize: bool = False):
    """Initialize a Dataset repository with the given name

        Args:
            data_dir (str):
                Repository location. This will be scanned for datasets.
            initialize (bool):
                If data_dir doesn't exist, pass `initialize=True` to create one.
        """
    self._data_dir = data_dir
    if tf.io.gfile.exists(self._data_dir):
      self._versions = scan_versions(self._data_dir)
      logger.info(f'Found existing repository: {self._data_dir}')
      logger.info(f'Dataset versions: {self._versions}')
    else:
      logger.info(f'{self._data_dir} does not exist.')
      if initialize:
        tf.io.gfile.makedirs(self._data_dir)
        with open(tf.io.gfile.join(self.location, _INDEX_FILENAME), 'w') as f:
          f.write('{}')
        logger.info(f'Initialized empty repository at {self._data_dir}')
      else:
        raise Exception(
            f'Cannot proceed, path does not exist (received `initialize={initialize}`)'
        )

  @property
  def location(self) -> str:
    return self._data_dir

  @property
  def index_path(self) -> str:
    return tf.io.gfile.join(self.location, _INDEX_FILENAME)

  @property
  def versions(self) -> List[str]:
    """Returns the list of verified dataset versions in this repository"""
    return self._versions

  def get_builder(self,
                  versions: Optional[List[str]] = None
                 ) -> tfds.core.DatasetBuilder:
    """Returns a DatasetBuilder for specified versions of dataset.

        If `None` provided, builder is constructed for all versions of the dataset.
        """
    if versions is None:
      versions = self._versions

    # Check validity
    if not isinstance(versions, list):
      raise Exception(f'Expected a list of versions, received {type(versions)}')

    for ver in versions:
      if ver not in self._versions:
        raise ValueError(
            f'Unknown version [{ver}]. Available versions {self._versions}')

    # Append data_dir to paths
    logger.info(
        f'Creating builder for versions: {versions}, from {self._data_dir}')

    return tfds.builder_from_directories(
        [tf.io.gfile.join(self._data_dir, v) for v in versions])

  def purge(self, version: str) -> None:
    """Purge the specified dataset version from the repository"""
    logger.info(f'Version to purge: {version}')
    path = tf.io.gfile.join(self._data_dir, version)
    if tf.io.gfile.exists(path):
      tf.io.gfile.rmtree(path)
      logger.info(f'Deleted: {path}')
    else:
      logger.warn(
          f'Dataset version {version} does not exist, will continue execution')


def scan_versions(root_dir):
  """
    Returns list of dataset versions in `root_dir`
    by checking the presence of `root_dir/<ver>/dataset_info.json`
    """
  contents = tf.io.gfile.listdir(root_dir)
  # Find dirs within root_dir/*
  dirs = [
      path for path in contents
      if tf.io.gfile.isdir(tf.io.gfile.join(root_dir, path))
  ]
  # Dirs within root_dir/ that contain dataset_info.json are valid datasets
  versions = [
      dirname for dirname in dirs if tf.io.gfile.exists(
          tf.io.gfile.join(root_dir, dirname, _DATASET_CONFIG))
  ]
  return versions
