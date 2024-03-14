# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build
.PHONY: run_classification down_classification
.PHONY: run_instance_segmentation down_instance_segmentation
.PHONY: run_object_detection down_object_detection
.PHONY: compose_down_all

DOCKER_COMPOSE ?= docker-compose.yml
PIPELINE_COUNT ?= 1
DEPENDENCY_COMPOSE ?= add_dependent_services.yml

build:
	docker build --tag python-demo:dev -f Dockerfile .

gen_classification:
	@PIPELINE_COUNT=${PIPELINE_COUNT} docker compose -f $(DEPENDENCY_COMPOSE) -f add_replicas_classification.yml convert > $(DOCKER_COMPOSE)

run_classification: gen_classification
	@PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f $(DOCKER_COMPOSE) up -d

down_classification:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f add_replicas_classification.yml down

gen_instance_segmentation:
	@PIPELINE_COUNT=${PIPELINE_COUNT} docker compose -f $(DEPENDENCY_COMPOSE) -f add_replicas_instance_segmentation.yml convert > $(DOCKER_COMPOSE)

run_instance_segmentation: gen_instance_segmentation
	@PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f $(DOCKER_COMPOSE) up -d

down_instance_segmentation:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f add_replicas_instance_segmentation.yml down

gen_object_detection:
	@PIPELINE_COUNT=${PIPELINE_COUNT} docker compose -f $(DEPENDENCY_COMPOSE) -f add_replicas_object_detection.yml convert > $(DOCKER_COMPOSE)

run_object_detection: gen_object_detection
	@PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f $(DOCKER_COMPOSE) up -d

down_object_detection:
	@PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f add_replicas_object_detection.yml down

compose_down_all: down_classification down_instance_segmentation down_object_detection
	@docker compose -f $(DEPENDENCY_COMPOSE) down
