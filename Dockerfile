#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

#FROM intel/dlstreamer:2022.2.0-ubuntu20-gpu815 as base
FROM intel/dlstreamer:2023.0.0-ubuntu22-gpu682-dpcpp as base
USER root

FROM base as build-default
COPY ./requirements.txt /requirements.txt
RUN pip3 install --upgrade pip --no-cache-dir -r /requirements.txt
WORKDIR /
COPY /extensions /home/pipeline-server/extensions
COPY /framework-pipelines /home/pipeline-server/framework-pipelines
COPY /models /home/pipeline-server/models
COPY entrypoint.sh /script/entrypoint.sh
RUN mkdir /tmp/results