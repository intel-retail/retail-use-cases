#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Configure data extraction/loading primitives"""
from typing import Callable, Optional

import tensorflow as tf
import tensorflow_datasets as tfds


class Decoders:
  """
  Decoders.SIMPLE_DECODER: Simple Image-Label only decoder.

  Performant, memory-efficient, allows caching maximum data onto memory

  About:
    - This decoder only loads `image` and `label` elements
    from the TFRecord dataset.
    - `image` decoding is skipped, i.e., loaded as a raw string.
    - You must decode `image` to `tf.tensor` using a `dataset.map()` function
  """
  SIMPLE_DECODER = tfds.decode.PartialDecoding(
      {
          'image': True,
          'label': True,
      },
      decoders={
          'image': tfds.decode.SkipDecoding(),
      })

"""Configure data augmentation while training"""
IMG_AUGMENT_LAYERS = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal')
], name='augment_layers')