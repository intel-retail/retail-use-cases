#!/bin/bash

# Copyright (c) 2022 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# This script will download BiT models and create OpenVINO IRs.
#

# Download BiT models.

wget https://tfhub.dev/google/bit/m-r50x1/1?tf-hub-format=compressed -O bit_m_r50x1_1.tar.gz
mkdir -p bit_m_r50x1_1 && tar -xvf bit_m_r50x1_1.tar.gz -C bit_m_r50x1_1

# Run Model Optimization and create OpenVINO IRs
mo --framework tf \
--saved_model_dir ./bit_m_r50x1_1 \
--output_dir ov_irs/bit_m_r50x1_1/

# copy desire models into volume mount /result folder, and use mounted volume to have downloaded model files into our structure of bit models
cp /ov_irs/bit_m_r50x1_1/saved_model.bin /result/bit_64.bin
cp /ov_irs/bit_m_r50x1_1/saved_model.xml /result/bit_64.xml
