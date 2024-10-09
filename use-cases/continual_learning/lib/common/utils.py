#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

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
