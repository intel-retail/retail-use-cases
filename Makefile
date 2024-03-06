# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
.PHONY: build-all download-models clean-models build-profile-launcher update-submodules build-ovms_server
.PHONY: download-sample-media
.PHONY: build-gst run-gst down-gst
.PHONY: build-grpc_python list-grpc-python-model-names run-grpc_python down-grpc_python
.PHONY: build-capi_yolov5 run-capi_yolov5 down-capi_yolov5
.PHONY: build-capi_yolov5_ensemble run-capi_yolov5_ensemble down-capi_yolov5_ensemble
.PHONY: build-capi_face_detection run-capi_face_detection down-capi_face_detection
.PHONY: build-all-capis
.PHONY: clean-results

USECASES= \
			demos \
			dlstreamer \
			grpc_go \
			grpc_python \
			gst_capi \
			ovms_server

.PHONY: $(USECASES)

build-all:
		for repo in ${USECASES}; do \
			echo use-cases/$$repo; \
			cd use-cases/$$repo; \
			make build || exit 1; \
			cd ../..; \
		done

download-models:
	./models-downloader/downloadOVMSModels.sh

clean-models:
	@find ./models/ -mindepth 1 -maxdepth 1 -type d -exec rm -r {} \;

clean-results:
	@find ./results/ -maxdepth 1 -type f ! -name "result.txt" -exec sudo rm -r {} \;

build-profile-launcher:
	@cd ./core-services && $(MAKE) build-profile-launcher

build-ovms_server:
	@cd ./use-cases/ovms_server && $(MAKE) build

update-submodules:
	@git submodule update --init --recursive
	@git submodule update --remote --merge

download-sample-media:
	@cd ./performance-tools/benchmark-scripts && ./download_sample_videos.sh

build-gst:
	@cd ./use-cases/dlstreamer && $(MAKE) --no-print-directory build

prepare-inputs: update-submodules download-models download-sample-media

run-gst: prepare-inputs
	@cd ./use-cases/dlstreamer && $(MAKE) --no-print-directory run

down-gst:
	@cd ./use-cases/dlstreamer && $(MAKE) --no-print-directory down

build-grpc_python: build-ovms_server
	@cd ./use-cases/grpc_python && $(MAKE) --no-print-directory build

list-grpc-python-model-names:
	@cd ./use-cases/grpc_python && $(MAKE) --no-print-directory model-names

run-grpc_python: prepare-inputs list-grpc-python-model-names
	@cd ./use-cases/grpc_python && $(MAKE) --no-print-directory run

down-grpc_python:
	@cd ./use-cases/grpc_python && $(MAKE) --no-print-directory down

build-capi_yolov5:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory build_capi_yolov5

run-capi_yolov5:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory DOCKER_COMPOSE=docker-compose_replicas_capi_yolov5.yml run_capi_yolov5

down-capi_yolov5:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory DOCKER_COMPOSE=docker-compose_replicas_capi_yolov5.yml down_capi_yolov5

build-capi_yolov5_ensemble:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory build_capi_yolov5_ensemble

run-capi_yolov5_ensemble:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory DOCKER_COMPOSE=docker-compose_replicas_capi_yolov5_ensemble.yml run_capi_yolov5_ensemble

down-capi_yolov5_ensemble:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory DOCKER_COMPOSE=docker-compose_replicas_capi_yolov5_ensemble.yml down_capi_yolov5_ensemble

build-capi_face_detection:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory build_face_detection

run-capi_face_detection:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory DOCKER_COMPOSE=docker-compose_replicas_capi_face_detection.yml run_capi_face_detection

down-capi_face_detection:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory DOCKER_COMPOSE=docker-compose_replicas_capi_face_detection.yml down_capi_face_detection

build-all-capis:
	@cd ./use-cases/gst_capi && $(MAKE) --no-print-directory build
