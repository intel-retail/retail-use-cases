FROM openvino/model_server:2023.1-gpu

USER root
RUN if [ -n "$HTTP_PROXY" ] ; then  echo "Acquire::http::Proxy \"$HTTP_PROXY\";" >  /etc/apt/apt.conf; fi
RUN apt-get update -y || true; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    jq

COPY config.json /config/config.json
COPY entrypoint.sh /script/entrypoint.sh

USER ovms
ENTRYPOINT ["/script/entrypoint.sh"]