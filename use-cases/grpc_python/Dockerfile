#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

FROM ubuntu:20.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN if [ -n "$HTTP_PROXY" ]; then echo "Acquire::http:Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf; fi

RUN apt-get update -y || true; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/openvinotoolkit/model_server
WORKDIR /model_server/client/python/kserve-api/samples
RUN pip3 install -r requirements.txt; pip3 install opencv-python
WORKDIR /app
COPY . .
RUN chmod +x /app/entrypoint.sh
