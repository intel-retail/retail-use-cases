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

"""Common utilities"""

import logging
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler


def progress(iterable, **kwargs):
  """Formatted progressbar"""
  return tqdm(iterable,
              bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
              **kwargs)


def setup_loggers(log_level=logging.INFO) -> None:
  """Configure loggers."""
  root = logging.getLogger()
  root.handlers = []  # workaround to remove absl/existing loggers
  root.setLevel(log_level)
  console = Console(width=160)
  handler = RichHandler(console=console)
  formatter = logging.Formatter('%(message)s')
  handler.setFormatter(formatter)
  root.addHandler(handler)
