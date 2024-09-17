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

"""Tools to query/manipulate tf.data.Dataset"""
from logging import getLogger
from typing import Any, Callable, Sequence, Tuple, Optional, Mapping, List

import numpy as np
import tensorflow as tf

from lib.common.utils import progress

logger = getLogger(__name__)


def preprocess_imagenet(mode: str = 'torch') -> Callable:
  """Preprocess function applied for ImageNet.

  Args:
      mode (str, optional):
        One of "caffe", "tf" or "torch". Defaults to "tf".
        
        caffe:
          Converts the images from RGB to BGR,
          then will zero-center each color channel with
          respect to the ImageNet dataset, without scaling.
        tf:
          Scales pixels between -1 and 1, sample-wise.
        torch:
          Scales pixels between 0 and 1 and then
          will normalize each channel with respect
          to the ImageNet dataset.

  Returns:
      Callable:
        Callable function that applies preprocessing on `image` tensor
        of each example. Each example must be an (x, y) tuple.
  """
  @tf.function
  def _preprocess(x, y):
    return (tf.keras.applications.imagenet_utils.preprocess_input(x,
                                                                  mode=mode), y)

  return _preprocess


def decode_example(resize: Tuple[int, int]) -> Callable:
  """Decodes and resizes the image"""

  def _decode_and_resize(example):
    image = tf.io.decode_image(example['image'],
                               channels=3,
                               expand_animations=False)
    example['image'] = tf.image.resize(image, resize)
    return example

  return _decode_and_resize


def as_tuple(x: str, y: str):
  """
  Returns values from each sample as a supervised tuple.
  Provide `x`, `y` as keys to be accessed from each sample.

  Example usage: `dataset = dataset.map(as_tuple(x='image', y='label'))`

  This is typically used in the last step in the data pipeline
  before topping off with batch/prefetch.
  """
  return lambda example: (example[x], example[y])


def get_label_distribution(ds: tf.data.Dataset) -> Mapping[int, int]:
  """Returns class distribution of requested dataset"""
  labels = list(ds.map(lambda s: s['label']).as_numpy_iterator())
  class_ids, counts = np.unique(labels, return_counts=True)
  dist = dict(zip(class_ids, counts))
  return dist


def filter_classes(y, allowed_classes: list):
  """
    Returns `True` if `y` belongs to `allowed_classes` list else `False`
    Example usage:
        dataset.filter(lambda s: filter_classes(s['label'], [0,1,2])) # as dict
        dataset.filter(lambda x, y: filter_classes(y, [0,1,2])) # as_supervised
    """
  allowed_classes = tf.constant(allowed_classes)
  isallowed = tf.equal(allowed_classes, tf.cast(y, allowed_classes.dtype))
  reduced_sum = tf.reduce_sum(tf.cast(isallowed, tf.float32))
  return tf.greater(reduced_sum, tf.constant(0.))


def shard_sequence(seq: Sequence[Any], n) -> List[List]:
  """Shard the given sequence into partitions of fixed size `n`"""
  if n in (-1, 0):
    return [seq]
  return [seq[i:i + n] for i in range(0, len(seq), n)]


def get_class_weights(dataset) -> list:
  """
    Returns list of floats (a.k.a weights) to be applied
    to individual classes in the dataset
    """
  dist = get_label_distribution(dataset)
  num_classes = len(dist.keys())
  total = sum(dist.values())
  class_wts = []
  for id in range(num_classes):
    if dist.get(id, False):
      weight = (1 / dist[id]) * total / num_classes
    else:
      weight = 0
    class_wts.append(weight)
  return class_wts


def gather_replay_examples(
    ds: tf.data.Dataset,
    examples_per_class: Optional[int] = 10) -> tf.data.Dataset:
  """Generates a replay dataset by picking specified number of examples per class.

    Note:
        - Pass raw, string-encoded image dataset for faster I/O and storage efficiency
        - If `examples_per_class` is greater than the total available samples
        for any given class, all samples will be picked.

    Args:
        ds (`tf.data.Dataset`): Source dataset to sample from.
        examples_per_class (Optional[int], optional): Defaults to 10.
        as_supervised (Optional[bool], optional):
            ElementSpec of an example in the source dataset.
            Pass `True` if each sample is an (x, y) tuple, `False` if it is a dict.
            Defaults to True.

    Returns:
        tf.data.Dataset: A dataset with elements sampled from source `ds`
    """
  if 'label' not in ds.element_spec:
    raise TypeError('Input dataset should have label keyed as `label`.')

  labels = ds.map(lambda s: s['label'])

  # Labels to iterate over
  unique_labels = sorted(list(labels.unique().as_numpy_iterator()))

  # Take `examples_per_class` from each subset
  logger.info('Taking {} examples per class (num_classes={})'.format(
      examples_per_class, len(unique_labels)))

  replay_ds = None
  for label in progress(unique_labels, desc='Progress'):
    subset = ds.filter(lambda s: s['label'] == label)
    _cache = subset.take(examples_per_class)
    # Extend the replay dataset for each class
    if replay_ds is None:
      replay_ds = _cache
    else:
      replay_ds = replay_ds.concatenate(_cache)

  n_examples = sum([1 for _ in replay_ds])
  logger.info(f'Captured {n_examples} examples as exact replay.')
  return replay_ds
