# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

from logging import getLogger

import tensorflow as tf

logger = getLogger(__name__)


class SLDA(tf.keras.Model):
  """
  Lifelong machine learning with deep streaming Linear Discriminant Analysis.
  TL Hayes et. al. (CVPR'20): https://arxiv.org/abs/1909.01520

  LDA, at its simplest, is a linear model that learns simple n-dim multivariate
  Gaussian for each class with combined single covariance matrix.

  It can be interpreted as a Generative Model over **features** rather than high
  dimensional input images. Features can be extracted using a backbone
  pretrained on ImageNet, for instance.

  Example usage:
  ```python
  # Extract features from images using a backbone of your choice
  train_features = backbone_model(train_images)
  test_features = backbone_model(test_images)

  # LDA is a linear model over features as input, mapping to `num_classes` output classes.
  # Each feature could be `n_components` dimensional.
  slda_model = SLDA(n_components=512, num_classes=10)
  slda_model.compile(metrics=['accuracy'])

  # SLDA updates mean/covariance 1 sample at a time during training
  slda_model.fit(train_features.batch(1))

  # Evaluate over test dataset (any batch size works)
  slda_model.evaluate(test_features.batch(64))
  ```
  """

  def __init__(self,
               n_components: int,
               num_classes: int,
               shrinkage: float = 1e-4):
    """Instantiate an LDA model.

    Args:
        n_components (int):
          Input 1D feature dimension size.
        num_classes (int):
          Total output classes.
        shrinkage (float, optional):
          Shrinkage regularization factor. Defaults to 1e-4.
    """
    super(SLDA, self).__init__()

    # Parameters
    self.means = tf.Variable(
        initial_value=tf.zeros((num_classes, n_components)),
        trainable=False,
    )
    self.counts = tf.Variable(initial_value=tf.zeros(num_classes, tf.float32),
                              trainable=False)
    self.sigma = tf.Variable(initial_value=tf.zeros(
        (n_components, n_components)),
                             trainable=False)
    self.Lambda = tf.Variable(initial_value=tf.zeros_like(self.sigma),
                              trainable=False)
    self.shrinkage = shrinkage
    self._steps = tf.Variable(initial_value=0)
    self._trigger_update = tf.Variable(initial_value=True, trainable=False)

  def fit(self, X, **kwargs):
    if isinstance(X, tf.data.Dataset):
      (x, _) = next(iter(X))
      if x.shape[0] > 1:
        raise Exception(
            'batch>1 for training dataset is not supported (expected batch=1)')
      super().fit(X, **kwargs)

  def train_step(self, data):
    """Update mean/covariance for the given (x,y) pair"""
    # Unpack
    x, y = data

    # Calculate scatter
    x_minus_mu = (x - tf.gather(self.means, y))
    scatter = tf.matmul(tf.transpose(x_minus_mu, [1, 0]), x_minus_mu)
    delta = scatter * tf.cast(self._steps / (self._steps + 1), tf.float32)

    # Update means, counts, sigma
    self.sigma.assign((tf.cast(self._steps, tf.float32) * self.sigma + delta) /
                      tf.cast(self._steps + 1, tf.float32))
    self.means.assign(
        tf.tensor_scatter_nd_add(self.means, [y],
                                 x_minus_mu / (tf.gather(self.counts, y) + 1)))
    self.counts.assign(tf.tensor_scatter_nd_add(self.counts, [y], [1]))
    self._trigger_update.assign(True)
    self._steps.assign_add(1)

    history = dict()
    return history

  def test_step(self, data):
    x, y = data
    y_pred = self(x)
    self.compiled_metrics.update_state(y, y_pred)
    history = {m.name: m.result() for m in self.metrics}
    return history

  def call(self, x):
    """Inference step, update inverse if not done (once)"""
    if self._trigger_update:
      reg_sigma = (1 - self.shrinkage) * self.sigma + self.shrinkage * tf.eye(
          tf.shape(self.sigma)[0])
      self.Lambda.assign(tf.linalg.pinv(reg_sigma))
      self._trigger_update.assign(False)

    # Forward pass
    m_T = tf.transpose(self.means, [1, 0])
    W = tf.matmul(self.Lambda, m_T)
    b = -0.5 * tf.reduce_sum(m_T * W, axis=0)
    logits = tf.matmul(x, W) + b
    return logits