# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build run down

DOCKER_COMPOSE ?= docker-compose_grpc_go.yml
PIPELINE_COUNT ?= 1

build:
	docker build --tag grpc_go:dev .

run:
	PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f $(DOCKER_COMPOSE) up -d

down:
	PIPELINE_COUNT=$(PIPELINE_COUNT) docker compose -f $(DOCKER_COMPOSE) down
