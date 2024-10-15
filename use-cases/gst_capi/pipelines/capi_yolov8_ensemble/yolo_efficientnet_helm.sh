#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

PIPELINE_PROFILE="${PIPELINE_PROFILE:=capi-yolov8-ensemble}"
DEVICE="${DEVICE:=CPU}"

# User configured parameters
if [ -z "$INPUT_TYPE" ]
then
	echo "INPUT_TYPE is required"
	exit 1
	#INPUT_TYPE="FILE_H264"
	#INPUT_TYPE="RTSP_H265"
fi

if [ -z "$INPUTSRC" ]
then
	echo "INPUTSRC is required"
	exit 1
	#INPUTSRC="sample-video.mp4"
	#INPUTSRC="rtsp://127.0.0.1:8554/camera_0"
fi

CODEC_TYPE=0
if [ "$INPUT_TYPE" == "FILE_H264" ] || [ "$INPUT_TYPE" == "RTSP_H264" ]
then
	CODEC_TYPE=1
elif [ "$INPUT_TYPE" == "FILE_H265" ] || [ "$INPUT_TYPE" == "RTSP_H265" ]
then
	CODEC_TYPE=0
fi

if [ -z "$USE_VPL" ]
then
	USE_VPL=0
fi

if [ -z "$RENDER_MODE" ]
then
	RENDER_MODE=0
fi

if [ -z "$RENDER_PORTRAIT_MODE" ]
then
	RENDER_PORTRAIT_MODE=0
fi

# DEBUGGING prints:
env
ls -al /tmp/
ls -al /tmp/results/
ls -al /app/gst-ovms/pipelines/yolov8_ensemble/
echo "libva: /usr/lib/x86_64-linux-gnu/dri/ "
ls -al /usr/lib/x86_64-linux-gnu/dri/
ls -al /usr/lib/
ls -al /usr/lib/x86_64-linux-gnu/
ls -al /usr/lib/x86_64-linux-gnu/libva*.so*
ls -al /usr/lib/x86_64-linux-gnu/libva-drm*.so*

# generate unique container id based on the date with the precision upto nano-seconds
cid=$(date +%Y%m%d%H%M%S%N)
echo "cid: $cid"
echo "PIPELINE_PROFILE: $PIPELINE_PROFILE  DEVICE: $DEVICE"

echo "DC: $DC INPUTSRC: $INPUTSRC USE_VPL: $USE_VPL RENDER_MODE: $RENDER_MODE RENDER_PORTRAIT_MODE: $RENDER_PORTRAIT_MODE"
echo "CODEC_TYPE: $CODEC_TYPE WINDOW_WIDTH: $WINDOW_WIDTH WINDOW_HEIGHT: $WINDOW_HEIGHT DETECTION_THRESHOLD: $DETECTION_THRESHOLD"

appExitCode=0
# Direct console output
if [ "$DC" != 1 ]
then
	/app/gst-ovms/pipelines/yolov8_ensemble/capi_yolov8_ensemble $INPUTSRC $USE_VPL $RENDER_MODE $RENDER_PORTRAIT_MODE $CODEC_TYPE $WINDOW_WIDTH $WINDOW_HEIGHT $DETECTION_THRESHOLD 2>&1 | tee >/tmp/results/r"$cid"_"$PIPELINE_PROFILE".jsonl >(stdbuf -oL sed -n -e 's/^.*FPS: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PIPELINE_PROFILE".log)
	appExitCode=$?
	sleep 10
	ls -al /app/gst-ovms/pipelines/yolov8_ensemble/config-yolov8.json
	ls -al /models
	ls -al /dev/dri
	cat /tmp/results/r"$cid"_"$PIPELINE_PROFILE".jsonl
else
	/app/gst-ovms/pipelines/yolov8_ensemble/capi_yolov8_ensemble $INPUTSRC $USE_VPL $RENDER_MODE $RENDER_PORTRAIT_MODE $CODEC_TYPE $WINDOW_WIDTH $WINDOW_HEIGHT $DETECTION_THRESHOLD
	appExitCode=$?
fi

if [ $appExitCode != 0 ]; then
	echo "capi yolov8 ensemble failed with exit code=$appExitCode"
	exit 1
fi

echo "end of script..."
# make forever-loop so it is easy to debug into docker exec -it in case there is error
# you can always use "kubectl delete -f kubernetes" to terminate this container
while true
do
	sleep 1
done
