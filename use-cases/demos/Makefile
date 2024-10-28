# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build
.PHONY: run_classification down_classification
.PHONY: run_instance_segmentation down_instance_segmentation
.PHONY: run_object_detection down_object_detection
.PHONY: run_pytorch_object_detection down_pytorch_object_detection
.PHONY: compose_down_all
.PHONY: render

DOCKER_COMPOSE ?= docker-compose.yml
PIPELINE_COUNT ?= 1
DEPENDENCY_COMPOSE ?= add_dependent_services.yml
RETAIL_USE_CASE_ROOT ?= ../..
RESULTS_DIR ?= ../../results

build:
	docker build --tag python-demo:dev -f Dockerfile .

build-pytorch:
	docker build --tag pytorch-demo:dev -f Dockerfile-pytorch .

render: 
	xhost +local:docker

gen_classification:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DEPENDENCY_COMPOSE) -f add_demo_classification.yml convert > $(DOCKER_COMPOSE)

run_classification: gen_classification
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DOCKER_COMPOSE) up -d

down_classification:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f add_demo_classification.yml down

gen_instance_segmentation:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DEPENDENCY_COMPOSE) -f add_demo_instance_segmentation.yml convert > $(DOCKER_COMPOSE)

run_instance_segmentation: gen_instance_segmentation
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DOCKER_COMPOSE) up -d

down_instance_segmentation:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f add_demo_instance_segmentation.yml down

gen_object_detection:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DEPENDENCY_COMPOSE) -f add_demo_object_detection.yml convert > $(DOCKER_COMPOSE)

run_object_detection: gen_object_detection
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DOCKER_COMPOSE) up -d

down_object_detection:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f add_demo_object_detection.yml down

gen_pytorch_object_detection:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f add_demo_pytorch_object_detection.yml --env-file pytorch-yolov5-env.txt convert > $(DOCKER_COMPOSE)

run_pytorch_object_detection: gen_pytorch_object_detection render
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DOCKER_COMPOSE) up -d

down_pytorch_object_detection:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f add_demo_pytorch_object_detection.yml down

compose_down_all: down_classification down_instance_segmentation down_object_detection down_pytorch_object_detection
	@docker compose -f $(DEPENDENCY_COMPOSE) down
