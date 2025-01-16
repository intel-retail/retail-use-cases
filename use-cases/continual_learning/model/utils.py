# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any

import numpy as np
import tensorflow as tf


def extract_features(dataset: tf.data.Dataset, model: Any) -> tf.data.Dataset:
  """Extract feature embeddings from the model for each image in the dataset.

  Args:
      dataset (tf.data.Dataset):
        A `tf.data.Dataset` instance with raw tensor image keyed as `image` and `label` as label.
      model (Any):
        A callable of type `tf.keras.Sequential` or `tf.keras.Model` or equivalent that can take `image`
        batch as input and return feature embedding.

  Returns:
      A `tf.data.Dataset` instance with each sample being a `(feature_embedding, label)` tuple
  """
  features = model.predict(dataset, verbose=1)
  labels = np.array(list(dataset.map(lambda x, y: y).unbatch().as_numpy_iterator()))
  return tf.data.Dataset.from_tensor_slices({
    'image': features,
    'label': labels
  })
