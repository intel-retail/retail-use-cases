#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

# generate unique container id based on the date with the precision upto nano-seconds
cid=$(date +%Y%m%d%H%M%S%N)
echo "cid: $cid"

echo "USE_ULTRALYTICS: $USE_ULTRALYTICS"
echo "INTEL_OPTIMIZED: $INTEL_OPTIMIZED"

if [ $USE_ULTRALYTICS -eq 1 ]; 
then
 if [ $INTEL_OPTIMIZED -eq 1 ]; 
 then 
 PROFILE_NAME="demo_intel_ultralytics_pytorch_object_detection"
 python3 object_detection/pytorch-yolo/yolov5_ultralytics_object_detection.py \
-m "$MQTT_HOSTNAME" -p "$MQTT_PORT" -t "$MQTT_TOPIC" -i "$INPUT_SRC" -in | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*FPS: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)
 else
 PROFILE_NAME="demo_ultralytics_pytorch_object_detection"
 python3 object_detection/pytorch-yolo/yolov5_ultralytics_object_detection.py \
-m "$MQTT_HOSTNAME" -p "$MQTT_PORT" -t "$MQTT_TOPIC" -i "$INPUT_SRC" | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*FPS: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)
 fi
else
 PROFILE_NAME="demo_pytorch_object_detection"
 python3 object_detection/pytorch-yolo/yolov5_object_detection.py \
-m "$MQTT_HOSTNAME" -p "$MQTT_PORT" -t "$MQTT_TOPIC" -i "$INPUT_SRC" | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*FPS: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)
fi