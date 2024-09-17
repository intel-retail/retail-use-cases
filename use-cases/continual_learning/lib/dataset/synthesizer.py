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

"""Library for synthesizing Continual-Learning datasets"""
import collections
import logging
from typing import Any, List, Mapping, MutableMapping, Optional

import numpy as np
import tensorflow as tf

from lib.dataset.utils import filter_classes
from lib.dataset.utils import shard_sequence

logger = logging.getLogger(__name__)


def _convert_list_of_elems_to_tensor_slices(
    list_of_elems: List[Mapping[str, tf.Tensor]]) -> Mapping[str, tf.Tensor]:
  """Stacks individual elements in the mapping as a single big stacked tensor.

  Partitioned dataset elements are each described by a dict with key-value pairs.
  In order to convert this into a tf.data.Dataset, all values should be stacked
  as a tensor for each key.

  In essence, this is a conversion where if:
  ```python
  inp = [{'image': A, 'label': 6}, {'image': B, 'label': 3}]
  out = {'image': [A, B], 'label': [6, 3]}
  tf.data.Dataset.from_tensor_slices(out) == inp # True
  ```
  Args:
      list_of_elems (`List[Mapping[str, tf.Tensor]]`): List of tf.data.Dataset elements

  Returns:
      `Mapping[str, tf.Tensor]`: Stacked over key, tensor slice.
  """
  tensor_slices = collections.OrderedDict()
  for key in list_of_elems[0]:
    tensor_slices[key] = tf.stack([elem[key] for elem in list_of_elems])
  return tensor_slices


class _DirichletOverLabelsSynthesizer():
  """Backend class for function `synthesize_by_dirichlet_over_labels`"""

  def __init__(self,
               dataset: tf.data.Dataset,
               num_partitions: int,
               concentration_factor: float,
               seed: Optional[int] = None):
    if not isinstance(dataset.element_spec, Mapping):
      raise TypeError('Input dataset should have a Mapping type element_spec.')
    if 'label' not in dataset.element_spec:
      raise TypeError('Input dataset should have label keyed by `label`.')

    # Create random generator for ops
    self._rng = np.random.default_rng(seed)
    self._partition_ids = list(range(num_partitions))
    self._concentration_factor = concentration_factor
    self._element_spec = dataset.element_spec

    # Unpack dataset
    logger.info('Start unpacking the dataset.')
    self._dataset_list = list()
    for logging_cnt, elem in enumerate(dataset.as_numpy_iterator()):
      if logging_cnt % 10000 == 0:
        logging.info(f'Unpacking dataset, {logging_cnt} elems processed')
      self._dataset_list.append(elem)
    logger.info('Finish unpacking the dataset.')

    # Preprocessing
    self._elem_pools_by_label = self._build_elem_pools_by_label()
    self._partition_multinomials = self._sample_multinomial_of_all_partitions()

  def _build_elem_pools_by_label(self) -> Mapping[int, List[Any]]:
    """Build a pool of elements (by id) for each label

    Returns:
      A mapping with key pointing to the label and value to be the
      corresponding indices of the label in the original dataset.
    """
    elem_pools_by_label = collections.OrderedDict()

    for logging_cnt, element in enumerate(self._dataset_list):
      if logging_cnt % 10000 == 0:
        logger.info(
            f'Building elem pools by label, {logging_cnt} of {len(self._dataset_list)} elems processed.'
        )
      label = element['label']
      if label not in elem_pools_by_label:
        elem_pools_by_label[label] = list()

      elem_pools_by_label[label].append(element)

    map(self._rng.shuffle, elem_pools_by_label)
    return elem_pools_by_label

  def _compute_prior(self) -> Mapping[int, float]:
    """Compute the prior distribution for each label in the dataset"""
    prior = collections.OrderedDict()
    for label, pool in self._elem_pools_by_label.items():
      prior[label] = len(pool) / len(self._dataset_list)
    return prior

  def _sample_multinomial_of_all_partitions(
      self) -> Mapping[int, MutableMapping[Any, float]]:
    """Sample the multinomial for all partitions"""
    prior = self._compute_prior()

    partition_multinomials = collections.OrderedDict()

    for partition_id in self._partition_ids:
      multinomial = self._rng.dirichlet(self._concentration_factor *
                                        np.ones(len(prior)))
      # Seems to give stable DIR
      partition_multinomials[partition_id] = {
          label: prob
          for label, prob in zip(self._elem_pools_by_label, multinomial)
      }

    return partition_multinomials

  def _renormalize_multinomial(self, multinomial: MutableMapping[Any, float],
                               label_to_reset: Any):
    """Reset the exhausted label and renormalize multinomial in-place."""
    multinomial[label_to_reset] = 0.
    normalizer = sum(multinomial.values())
    for label in multinomial:
      multinomial[label] /= normalizer

  def _renormalize_multinomial_of_all_partitions(self, label):
    for partition_id in self._partition_ids:
      self._renormalize_multinomial(self._partition_multinomials[partition_id],
                                    label)

  def _sample_a_label_by_multinomial(self, multinomial: MutableMapping[Any,
                                                                       float]):
    """Sample a label from a multinomial distribution."""
    label_idx = self._rng.choice(range(len(multinomial)),
                                 p=list(multinomial.values()))
    return list(multinomial.keys())[label_idx]

  def build_partitioned_data(
      self, rotate_draw: bool) -> Mapping[int, tf.data.Dataset]:
    """Build a partitioned dataset by drawing samples from the multinomial"""
    samples_per_partition = len(self._dataset_list) // len(self._partition_ids)
    partition_pools = {i: list() for i in self._partition_ids}
    logging_cnt = 0

    def _draw_once(partition_id: int, logging_cnt: int):
      if logging_cnt % ((len(self._dataset_list) + 9) // 10) == 0:
        logger.info('Creating synthesized dataset, {} of {} processed.'.format(
            logging_cnt, len(self._dataset_list)))

      multinomial = self._partition_multinomials[partition_id]
      sampled_label = self._sample_a_label_by_multinomial(multinomial)
      sampled_elem = self._elem_pools_by_label[sampled_label].pop()
      partition_pools[partition_id].append(sampled_elem)

      # If label exhausted, renormalize multinomial of all partitions
      if not self._elem_pools_by_label[sampled_label]:
        self._renormalize_multinomial_of_all_partitions(sampled_label)

      return logging_cnt + 1

    if rotate_draw:
      # Sample for one partition -> move to next partition
      for _ in range(samples_per_partition):
        for partition_id in self._rng.permutation(self._partition_ids):
          logging_cnt = _draw_once(partition_id, logging_cnt)
    else:
      # Sample until entire partition is filled -> move to next partition
      for partition_id in self._rng.permutation(self._partition_ids):
        for _ in range(samples_per_partition):
          logging_cnt = _draw_once(partition_id, logging_cnt)

    # Build tensor_slices
    tensor_slices = {
        partition_id:
        _convert_list_of_elems_to_tensor_slices(partition_pools[partition_id])
        for partition_id in self._partition_ids
    }
    # Build datasets from tensors
    partitioned_datasets = {
        partition_id:
        tf.data.Dataset.from_tensor_slices(tensor_slices[partition_id])
        for partition_id in self._partition_ids
    }
    return partitioned_datasets


def synthesize_by_dirichlet_over_labels(
    dataset: tf.data.Dataset,
    num_partitions: int,
    concentration_factor: float = 0.01,
    use_rotate_draw: bool = True,
    seed: Optional[int] = 1,
) -> Mapping[Any, tf.data.Dataset]:
  """Construct a heterogeneously-partitioned dataset from the given `dataset`.

  Sampling based on Dirichlet distribution over categories.

  Args:
      dataset (tf.data.Dataset): Original dataset to be partitioned.
      num_partitions (int): Number of partitions to create.
      concentration_factor (float, optional): Heterogeneity factor.
        If closer to 0, each partition gets data from a few unique labels.
        If approaches infinity, each partition roughly represents the population average.
        Defaults to 1.0.
      use_rotate_draw (bool, optional): Whether to rotate the drawing clients.
        If `True`, each partition will draw a sample only once and
        then rotate to next random partition. If `False` each partition will draw all
        samples before moving on to the next partition. Defaults to True.
      seed (Optional[int], optional): Seed for random ops. Defaults to 1.

  Returns:
      dict: A dictionary mapping the `partition_id` (key) to its `tf.data.Dataset` (value)
  """
  synthesizer = _DirichletOverLabelsSynthesizer(
      dataset=dataset,
      num_partitions=num_partitions,
      concentration_factor=concentration_factor,
      seed=seed)
  return synthesizer.build_partitioned_data(rotate_draw=use_rotate_draw)


def synthesize_by_sharding_over_labels(
    dataset: tf.data.Dataset,
    num_partitions: int,
    shuffle_labels: bool = True,
    seed: Optional[int] = 0,
) -> Mapping[Any, tf.data.Dataset]:
  """Creates mutually-exclusive class partitioned dataset.

  Each partition of the dataset has unique labels with no overlap on others.

  Args:
      dataset (tf.data.Dataset):
        Original dataset to partition from.
      num_partitions (int, optional):
        Number of partitions. Defaults to 1 (entire dataset).
      shuffle_labels (Optional[bool], optional):
        Shuffle the order of labels globally before partitioning. Defaults to False.
      seed (Optional[int], optional):
        Shuffle seed value (integer). Only used if shuffle is enabled.

  Raises:
      TypeError: If `dataset` does not contain `label` keyed label.

  Returns:
      dict: Dictionary of `tf.data.Dataset` keyed by partition number.
  """

  if 'label' not in dataset.element_spec:
    raise TypeError('Input dataset should have label keyed as `label`.')

  if num_partitions == 1:
    logger.info('Single partition requested, returning as-is.')
    return {0: dataset}

  unique_labels = dataset.map(lambda s: s['label']).unique()
  classes = sorted(list(unique_labels.as_numpy_iterator()))
  partition_size = len(classes) // num_partitions
  remainder = len(classes) % num_partitions

  logger.info(f'Total classes {len(classes)}, '
              f'Classes per partition {partition_size}, '
              f'(num_partitions={num_partitions})')

  if shuffle_labels:
    np.random.default_rng(seed).shuffle(classes)

  # Calculate partitions
  partition_classes = [classes[i * partition_size:(i + 1) * partition_size] for i in range(num_partitions)]

  # Distribute remaining labels across partitions
  for i in range(remainder):
    partition_classes[i].append(classes[-(i+1)])

  partitioned_dataset = {
      i:
      dataset.filter(lambda s: filter_classes(s['label'], partition_classes[i]))
      for i in range(num_partitions)
  }
  return partitioned_dataset
