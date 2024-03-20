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

PROFILE_NAME="demo_instance_segmentation"

python3 instance_segmentation/python/instance_segmentation_demo.py -m localhost:"$GRPC_PORT"/models/instance-segmentation-security-1040 \
--label instance_segmentation/python/coco_80cl_bkgr.txt -i $INPUTSRC \
--adapter ovms -t 0.85 --show_scores --show_boxes --output_resolution 1280x720 $mqttArgs 2>&1  | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*fps: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)
