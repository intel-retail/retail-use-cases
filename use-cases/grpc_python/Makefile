# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build run down model-names

DOCKER_COMPOSE ?= docker-compose_grpc_python.yml
PIPELINE_COUNT ?= 1
# supported MODEL_NAME: instance-segmentation-security-1040, bit_64, or yolov5s
MODEL_NAME ?= instance-segmentation-security-1040
RETAIL_USE_CASE_ROOT ?= ../..
RESULTS_DIR ?= ../../results

build:
	docker build -t grpc_python:dev .

build-server:
	docker build -t ovms-server:dev -f pipeline-server/Dockerfile ../..

model-names:
	@echo
	@echo "For grpc_python, supported MODEL_NAME value: instance-segmentation-security-1040, bit_64, or yolov5s"
	@echo "the default value for MODEL_NAME is instance-segmentation-security-1040 if not provided."
	@echo
	@echo "example: MODEL_NAME=yolov5s make run-grpc_python"
	@echo

run:
	PIPELINE_COUNT=$(PIPELINE_COUNT) MODEL_NAME=$(MODEL_NAME) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DOCKER_COMPOSE) up -d

down:
	PIPELINE_COUNT=$(PIPELINE_COUNT) MODEL_NAME=$(MODEL_NAME) RETAIL_USE_CASE_ROOT="$(RETAIL_USE_CASE_ROOT)" RESULTS_DIR="$(RESULTS_DIR)" docker compose -f $(DOCKER_COMPOSE) down

build-minikube:
	minikube start
	minikube image build -t grpc_python:dev -f Dockerfile .
	minikube image build -t ovms-server:dev -f use-cases/grpc_python/pipeline-server/Dockerfile ../../

run-minikube-demo: build-minikube helm-convert install-helm
	
stop-minikube-demo:
	kubectl delete -f object-detection-service-distributed-grpc
	minikube stop

helm-convert:
	kompose -f docker-compose_grpc_python.yml convert -o object-detection-service-distributed-grpc -c

install-helm:
	helm install -f object-detection-service-distributed-grpc/Chart.yaml dls object-detection-service-distributed-grpc

uninstall-helm:
	helm uninstall dls