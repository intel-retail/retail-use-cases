#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

FROM openvino/model_server:2023.1-gpu

USER root
RUN if [ -n "$HTTP_PROXY" ] ; then  echo "Acquire::http::Proxy \"$HTTP_PROXY\";" >  /etc/apt/apt.conf; fi
RUN apt-get update -y || true; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    jq \
    wget

COPY use-cases/grpc_python/pipeline-server/config.json /config/config.json
COPY use-cases/ovms_server/entrypoint.sh /script/entrypoint.sh
RUN chmod +x /script/entrypoint.sh
COPY use-cases/grpc_python/pipeline-server/downloadModels.sh /home/pipeline-server/models/downloadModels.sh
WORKDIR /home/pipeline-server/models
RUN sh downloadModels.sh
WORKDIR /

USER ovms
ENTRYPOINT ["/script/entrypoint.sh"]
