# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build build_face_detection build_capi_yolov5 build_capi_yolov5_ensemble
.PHONY: run_capi_yolov5 down_capi_yolov5
.PHONY: run_capi_yolov5_ensemble down_capi_yolov5_ensemble
.PHONY: run_capi_face_detection down_capi_face_detection

# initial default values
OVMS_CPP_DOCKER_IMAGE ?= openvino/model_server
OVMS_CPP_IMAGE_TAG ?= latest
DGPU_TYPE ?= arc # arc|flex
DOCKER_COMPOSE ?= docker-compose.yml
PIPELINE_COUNT ?= 1
CAMERA_SIMULATOR_COMPOSE ?= add_camera-simulator.yml
RETAIL_USE_CASE_ROOT ?= ../..
RESULTS_DIR ?= ../../results

# utility functions
define copy_ovms_config
	cd ../ovms_server && sudo cp ./config.json ../../models/$(1)
endef

define clean_ovms_config
	sudo rm -f ../../models/$(1) | true
endef


build: build_face_detection build_capi_yolov5 build_capi_yolov5_ensemble

build_face_detection:
	# Build CAPI docker image
	docker build $(NO_CACHE_OPTION) -f Dockerfile ../ \
		--build-arg http_proxy=$(HTTP_PROXY) \
		--build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		--build-arg DGPU_TYPE=$(DGPU_TYPE) \
		--build-arg PIPELINE_NAME=face_detection \
		--progress=plain \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi-gst-ovms-face_detection:$(OVMS_CPP_IMAGE_TAG)

build_capi_yolov5:
	# Build CAPI docker image
	docker build $(NO_CACHE_OPTION) -f Dockerfile ../ \
		--build-arg http_proxy=$(HTTP_PROXY) \
		--build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		--build-arg DGPU_TYPE=$(DGPU_TYPE) \
		--build-arg PIPELINE_NAME=capi_yolov5 \
		--progress=plain \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi-gst-ovms-capi_yolov5:$(OVMS_CPP_IMAGE_TAG)

build_capi_yolov5_ensemble:
	docker build $(NO_CACHE_OPTION) -f Dockerfile ../ \
		--build-arg http_proxy=$(HTTP_PROXY) \
		--build-arg https_proxy="$(HTTPS_PROXY)" \
		--build-arg no_proxy=$(NO_PROXY) \
		--build-arg BASE_IMAGE=ubuntu:22.04 \
		--build-arg DGPU_TYPE=$(DGPU_TYPE) \
		--build-arg PIPELINE_NAME=capi_yolov5_ensemble \
		--progress=plain \
		-t $(OVMS_CPP_DOCKER_IMAGE)-capi-gst-ovms-capi_yolov5_ensemble:$(OVMS_CPP_IMAGE_TAG)

gen_capi_yolov5:
	@PIPELINE_COUNT=${PIPELINE_COUNT} RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(CAMERA_SIMULATOR_COMPOSE) -f add_gst_capi_yolov5.yml convert > $(DOCKER_COMPOSE)

run_capi_yolov5: gen_capi_yolov5
	@$(call copy_ovms_config,config_capi_yolov5.json)
	PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) docker compose -f $(DOCKER_COMPOSE) up -d

down_capi_yolov5:
	PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(DOCKER_COMPOSE) down
	@$(call clean_ovms_config,config_capi_yolov5.json)

gen_capi_yolov5_ensemble:
	@PIPELINE_COUNT=${PIPELINE_COUNT} RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(CAMERA_SIMULATOR_COMPOSE) -f add_gst_capi_yolov5_ensemble.yml convert > $(DOCKER_COMPOSE)

run_capi_yolov5_ensemble: gen_capi_yolov5_ensemble
	@$(call copy_ovms_config,config_capi_yolov5_ensemble.json)
	PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(DOCKER_COMPOSE) up -d

down_capi_yolov5_ensemble:
	PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(DOCKER_COMPOSE) down
	@$(call clean_ovms_config,config_capi_yolov5_ensemble.json)

gen_capi_face_detection:
	@PIPELINE_COUNT=${PIPELINE_COUNT} RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(CAMERA_SIMULATOR_COMPOSE) -f add_gst_capi_face_detection.yml convert > $(DOCKER_COMPOSE)

run_capi_face_detection: gen_capi_face_detection
	@$(call copy_ovms_config,config_capi_face_detection.json)
	PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(DOCKER_COMPOSE) up -d

down_capi_face_detection:
	PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RESULTS_DIR=$(RESULTS_DIR) docker compose -f $(DOCKER_COMPOSE) down
	@$(call clean_ovms_config,config_capi_face_detection.json)
