#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

# https://github.com/openvinotoolkit/model_server/tree/main/client/python/kserve-api/samples
GRPC_PORT="${GRPC_PORT:=9001}"
OVMS_INIT_TIME_IN_SECOND="${OVMS_INIT_TIME_IN_SECOND:=10}"

# generate unique container id based on the date with the precision upto nano-seconds
cid=$(date +%Y%m%d%H%M%S%N)
echo "cid: $cid"

while :; do
    case $1 in
    --model_name)
        if [ "$2" ]; then
            DETECTION_MODEL_NAME=$2
            shift
        else
            error 'ERROR: "--model_name" requires an argument'
        fi
        ;;
    -?*)
        error "ERROR: Unknown option $1"
        ;;
    ?*)
        error "ERROR: Unknown option $1"
        ;;
    *)
        break
        ;;
    esac

    shift

done

echo "running grpc_python with GRPC_PORT= $GRPC_PORT, DETECTION_MODEL_NAME: $DETECTION_MODEL_NAME"

# this timing is to wait for ovms server models being ready
sleep $OVMS_INIT_TIME_IN_SECOND

PROFILE_NAME="grpc_python"

# Run the grpc python client
python3 ./grpc_python.py --input_src "$INPUTSRC" --grpc_address 127.0.0.1 --grpc_port "$GRPC_PORT" --model_name "$DETECTION_MODEL_NAME" \
2>&1  | tee >/tmp/results/r$cid"_$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*fps: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline$cid"_$PROFILE_NAME".log)
