#  Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

from logging import getLogger

import tensorflow as tf
import numpy as np

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

  def fit(self, X, y=None, **kwargs):
    """
    Custom fit method to handle NumPy arrays and process samples one by one.
    """
    # Just a check in case user passes tf.data.Dataset by mistake
    if isinstance(X, tf.data.Dataset):
        raise TypeError("SLDA.fit() expects NumPy arrays, not tf.data.Dataset")

    if X is None or y is None:
        raise ValueError("SLDA.fit() received None for X or y")

    if len(X) != len(y):
        raise ValueError(f"Mismatch: X has {len(X)} samples, y has {len(y)} labels")

    print(f"✅ SLDA.fit() received: X={X.shape}, y={y.shape}")
    
    # Process each sample individually
    for i in range(len(X)):
        # Convert single sample to TensorFlow tensors with correct shapes
        x_sample = tf.convert_to_tensor(X[i:i+1], dtype=tf.float32)  # Shape: (1, n_components)
        y_sample = tf.convert_to_tensor([y[i]], dtype=tf.int32)      # Shape: (1,)
        
        # Call train_step with single sample
        self.train_step((x_sample, y_sample))
    
    print(f"✅ SLDA training completed for {len(X)} samples")
    return self

  def train_step(self, data):
    """Update mean/covariance for the given (x,y) pair"""
    # Unpack
    x, y = data
    
    # Ensure y is a scalar for indexing
    y_scalar = y[0] if tf.rank(y) > 0 else y

    # Calculate scatter
    x_minus_mu = (x - tf.gather(self.means, y_scalar))
    scatter = tf.matmul(tf.transpose(x_minus_mu, [1, 0]), x_minus_mu)
    delta = scatter * tf.cast(self._steps / (self._steps + 1), tf.float32)

    # Update means, counts, sigma
    self.sigma.assign((tf.cast(self._steps, tf.float32) * self.sigma + delta) /
                      tf.cast(self._steps + 1, tf.float32))
    
    # Update means
    current_count = tf.gather(self.counts, y_scalar)
    mean_update = x_minus_mu / (current_count + 1)
    self.means.assign(
        tf.tensor_scatter_nd_add(self.means, [[y_scalar]], mean_update))
    
    # Update counts
    self.counts.assign(tf.tensor_scatter_nd_add(self.counts, [[y_scalar]], [1.0]))
    
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
    # Use tf.cond instead of Python if statement for graph compatibility
    def update_lambda():
      reg_sigma = (1 - self.shrinkage) * self.sigma + self.shrinkage * tf.eye(
          tf.shape(self.sigma)[0])
      self.Lambda.assign(tf.linalg.pinv(reg_sigma))
      self._trigger_update.assign(False)
      return self.Lambda
    
    def no_update():
      return self.Lambda
    
    # Update Lambda if needed
    tf.cond(self._trigger_update, update_lambda, no_update)

    # Forward pass
    m_T = tf.transpose(self.means, [1, 0])
    W = tf.matmul(self.Lambda, m_T)
    b = -0.5 * tf.reduce_sum(m_T * W, axis=0)
    logits = tf.matmul(x, W) + b
    return logits
    