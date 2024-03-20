#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

OVMS_INIT_TIME_IN_SECOND="${OVMS_INIT_TIME_IN_SECOND:=10}"
MQTT="${MQTT:=}"

if [ "$MQTT" != "" ]
then
	mqttArgs="--mqtt ${MQTT}"
fi

# generate unique container id based on the date with the precision upto nano-seconds
cid=$(date +%Y%m%d%H%M%S%N)
echo "cid: $cid"

# this timing is to wait for ovms server models being ready
sleep $OVMS_INIT_TIME_IN_SECOND

PROFILE_NAME="demo_classification"

python3 classification/python/classification_demo.py -m localhost:"$GRPC_PORT"/models/"$CLASSIFICATION_MODEL_NAME" \
	--label classification/python/labels/"$CLASSIFICATION_LABEL_FILE" -i $INPUTSRC \
	--adapter ovms --output_resolution "$CLASSIFICATION_OUTPUT_RESOLUTION" $mqttArgs 2>&1  | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*fps: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)
