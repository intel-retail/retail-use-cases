#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

modelPrecisionFP16INT8="FP16-INT8"
modelPrecisionFP32INT8="FP32-INT8"
modelPrecisionFP32="FP32"

REFRESH_MODE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --refresh)
            echo "running model downloader in refresh mode"
            REFRESH_MODE=1
            ;;
        *)
            echo "Invalid flag: $1" >&2
            exit 1
            ;;
    esac
    shift
done

MODEL_EXEC_PATH="$(dirname "$(readlink -f "$0")")"
modelDir="$MODEL_EXEC_PATH/../models"
mkdir -p "$modelDir"
cd "$modelDir" || { echo "Failure to cd to $modelDir"; exit 1; }

if [ "$REFRESH_MODE" -eq 1 ]; then
    # cleaned up all downloaded files so it will re-download all files again
    echo "In refresh mode, clean the existing downloaded models if any..."
    (
        cd "$MODEL_EXEC_PATH"/.. || echo "failed to cd to $MODEL_EXEC_PATH/.."
        make clean-models
    )
fi

pipelineZooModel="https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/"

# $1 model file name
# $2 download URL
# $3 model percision
getModelFiles() {
    # Get the models
    wget "$2"/"$3"/"$1"".bin" -P "$4"/"$1"/"$3"/1/
    wget "$2"/"$3"/"$1"".xml" -P "$4"/"$1"/"$3"/1/
}

# $1 model file name
# $2 download URL
# $3 json file name
# $4 process file name (this can be different than the model name ex. horizontal-text-detection-0001 is using horizontal-text-detection-0002.json)
# $5 precision folder
getProcessFile() {
    # Get process file
    wget "$2"/"$3".json -O "$5"/"$1"/"$3"/1/"$4".json
}

# custom yolov5s model downloading for this particular precision FP16INT8
downloadYolov5sFP16INT8() {
    yolov5s="yolov5s"
    yolojson="yolo-v5"
    modelType="/models"

    # checking whether the model file .bin already exists or not before downloading
    yolov5ModelFile="$modelType/$yolov5s/$modelPrecisionFP16INT8/$yolov5s.bin"
    if [ -f "$yolov5ModelFile" ]; then
        echo "yolov5s $modelPrecisionFP16INT8 model already exists in $yolov5ModelFile, skip downloading..."
    else
        echo "Downloading yolov5s $modelPrecisionFP16INT8 models..."
        # Yolov5s FP16_INT8
        getModelFiles $yolov5s $pipelineZooModel"yolov5s-416_INT8" $modelPrecisionFP16INT8 $modelType
        getProcessFile $yolov5s $pipelineZooModel"yolov5s-416_INT8" $yolojson $yolov5s $modelType
        echo "yolov5 $modelPrecisionFP16INT8 model downloaded in $yolov5ModelFile" $modelType
    fi
}

### Run custom downloader section below:
downloadYolov5sFP16INT8
