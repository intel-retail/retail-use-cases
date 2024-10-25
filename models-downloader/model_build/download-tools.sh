#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

# the changes in transformer compatiblity with openvino 2024 requires more right versions of dependencies
# eg. nncf 2.9+ and transformers 4.31+
pip install -q "openvino>=2024.4.0" "requests" "tqdm" "opencv-python" "transformers>=4.31" "onnx!=1.16.2" "torch>=2.1" "torchvision>=0.16" "ultralytics==8.3.0" onnx --extra-index-url https://download.pytorch.org/whl/cpu
pip install nncf>=2.9.0
