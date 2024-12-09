#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

# generate unique container id based on the date with the precision upto nano-seconds
cid=$(date +%Y%m%d%H%M%S%N)
echo "cid: $cid"

#USE_ULTRALYTICS for using ultralytics implementation of YOLO object detection
#Default set to 0, use torch hub implementation of YOLO object detection
echo "USE_ULTRALYTICS: $USE_ULTRALYTICS"

#INTEL_OPTIMIZED for using intel optimized packages of ultralytics implementation of YOLO object detection
#Default set to 0, torch hub implementation doesn't support intel optimized packages
echo "INTEL_OPTIMIZED: $INTEL_OPTIMIZED"

if [ $USE_ULTRALYTICS -eq 1 ]; 
then
 if [ $INTEL_OPTIMIZED -eq 1 ]; 
 then 
 PROFILE_NAME="demo_intel_ultralytics_pytorch_object_detection"
 DETECTION_SCRIPT="object_detection/pytorch-yolo/yolov5_ultralytics_object_detection.py -in "
 else
 PROFILE_NAME="demo_ultralytics_pytorch_object_detection"
 DETECTION_SCRIPT="object_detection/pytorch-yolo/yolov5_ultralytics_object_detection.py "
 fi
else
 PROFILE_NAME="demo_pytorch_object_detection"
 DETECTION_SCRIPT="object_detection/pytorch-yolo/yolov5_object_detection.py "
fi

python3 $DETECTION_SCRIPT \
-m "$MQTT_HOSTNAME" -p "$MQTT_PORT" -t "$MQTT_TOPIC" -i "$INPUT_SRC" | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*FPS: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)