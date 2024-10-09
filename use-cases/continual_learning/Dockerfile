#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#
FROM ubuntu:22.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN if [ -n "$HTTP_PROXY" ]; then echo "Acquire::http:Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf; fi

RUN apt-get update -y || true; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt requirements.txt
RUN python3 -m pip install -U pip
RUN pip install -r requirements.txt
