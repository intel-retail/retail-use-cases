# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Custom Loss functions for various model types"""
from logging import getLogger

import tensorflow as tf

logger = getLogger(__name__)

class SparsePolyLoss(tf.keras.losses.Loss):
  """
  Sparse label version of Poly1 CrossEntropy loss
  https://arxiv.org/pdf/2204.12511.pdf

  Outperforms standard CE loss on most datasets out of the box.
  """
  def __init__(self, epsilon=1.0):
    super().__init__()
    self.epsilon = epsilon

  def call(self, labels, logits):
    labels_onehot = tf.one_hot(labels, tf.shape(logits)[-1])
    pt = tf.reduce_sum(labels_onehot * tf.nn.softmax(logits), axis=-1)
    CE = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.reshape(labels, [-1]), logits)
    return CE + self.epsilon * (1-pt)

class EnergyLoss(tf.keras.losses.Loss):
  """Base class for Energy loss"""

  def __init__(self):
    super().__init__()

  def call(self, y_true, energy_pred):
    """Computes loss given positive y indices and energy prediction logits"""
    energy_pos = tf.gather(energy_pred,
                           batch_dims=-1,
                           indices=tf.expand_dims(y_true, 1))
    partition_estimate = tf.math.reduce_logsumexp(-energy_pred,
                                                  axis=1,
                                                  keepdims=True)
    loss = tf.reduce_mean(energy_pos + partition_estimate)
    return loss

  def sample(self, **kwargs):
    raise NotImplementedError(
        'This is a base class and must be inherited to implement a sampler.')


class BatchNegativePartitionEstimate(EnergyLoss):
  """
  Samples all negative examples from the current batch to
  compute partition estimate for each example.
  """
  def __init__(self):
    super().__init__()
    logger.info(f'EBM Loss: {type(self).__name__}')

  @staticmethod
  def sample(y, **kwargs):
    """Sample all negatives from current batch"""
    bs = tf.shape(y)[0]
    curr_classes = tf.unique(y)[0]
    y_pos = tf.vectorized_map(lambda t: tf.argmax(curr_classes == t), y)
    joint_targets = tf.reshape(tf.tile(curr_classes, [bs]), (bs, -1))
    return y_pos, joint_targets


class SingleNegativePartitionEstimate(EnergyLoss):
  """
  Samples one negative example from the current batch to
  compute partition estimate for each example.
  """
  def __init__(self):
    super().__init__()
    logger.info(f'EBM Loss: {type(self).__name__}')

  @staticmethod
  def sample(y, **kwargs):
    """Sample single negative per example from the batch"""
    # 0. Calculate shifts required in curr_classes such that each elem in y is at index 0
    curr_classes = tf.unique(y)[0]
    shifts = tf.vectorized_map(lambda t: tf.argmax(curr_classes == t), y)
    # 1. Roll curr_classes by these shifts to index 0.
    # 2. Discard first column. Remainder is the sampling pool
    pool = tf.vectorized_map(
        lambda s: tf.roll(curr_classes, shift=-s, axis=0)[1:], shifts)
    # 3. Shuffle each pool. Pick an elem at index (say 0) from each pool
    rnd_indices = tf.random.uniform(tf.shape(y),
                                    minval=0,
                                    maxval=tf.shape(curr_classes)[0] - 1,
                                    dtype=tf.int32)
    neg_samples = tf.gather(pool, rnd_indices, batch_dims=1)
    joint_targets = tf.stack([y, neg_samples], 1)
    y_pos = tf.zeros_like(y)
    return y_pos, joint_targets


class AllSeenPartitionEstimate(EnergyLoss):
  """
  Energies of all seen classes are considered to compute partition estimate.
  This is the logical equivalent of a CategoricalCrossentropy loss
  or standard SoftMax classifier.
  """
  def __init__(self):
    super().__init__()
    logger.info(f'EBM Loss: {type(self).__name__}')

  @staticmethod
  def sample(y, num_classes: int, **kwargs):
    """Use all seen classes as negatives (eqvt to SBC)"""
    bs = tf.shape(y)[0]
    curr_classes = tf.range(num_classes)
    joint_targets = tf.broadcast_to(tf.expand_dims(curr_classes, 0),
                                    (bs, num_classes))
    # y_pos == y
    return y, joint_targets
