# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build run down model-names

DOCKER_COMPOSE ?= docker-compose_replicas_grpc_python.yml
PIPELINE_COUNT ?= 1
# supported MODEL_NAME: instance-segmentation-security-1040, bit_64, or yolov5s
MODEL_NAME ?= instance-segmentation-security-1040

build:
	docker build -t grpc_python:dev .

model-names:
	@echo
	@echo "For grpc_python, supported MODEL_NAME value: instance-segmentation-security-1040, bit_64, or yolov5s"
	@echo "the default value for MODEL_NAME is instance-segmentation-security-1040 if not provided."
	@echo
	@echo "example: MODEL_NAME=yolov5s make run-grpc_python"
	@echo

run:
	PIPELINE_COUNT=$(PIPELINE_COUNT) MODEL_NAME=$(MODEL_NAME) docker compose -f $(DOCKER_COMPOSE) up -d

down:
	PIPELINE_COUNT=$(PIPELINE_COUNT) MODEL_NAME=$(MODEL_NAME) docker compose -f $(DOCKER_COMPOSE) down
