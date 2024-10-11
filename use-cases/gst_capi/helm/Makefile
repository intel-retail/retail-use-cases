# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build build_capi_yolov8_ensemble
.PHONY: minikube_build_capi_yolov8 helm_convert_capi_yolov8

# initial default values
DOCKER_COMPOSE ?= docker-compose.yml
PIPELINE_COUNT ?= 1
RENDER_MODE ?= 0
RETAIL_USE_CASE_ROOT ?= ../../../
RESULTS_DIR ?= /tmp/results

RELEASE_VERSION=3.2.0
YOLOV8_PACKAGE_NAME=capi-yolov8-ensemble

build: build_capi_yolov8_ensemble

build_capi_yolov8_ensemble:
	echo "Building CAPI YOLOv8 Ensemble container: HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY}"
	cd .. && docker build -t yolov8capiensemble:dev -f Dockerfile.yolov8 ./pipelines

minikube_build_capi_yolov8_ensemble: build
# clean up in case there is some left-over due to build error
	rm -rf ./pipelines || true
	cp -R ../pipelines .
	minikube start
	minikube image build -t yolov8capiensemble:dev -f Dockerfile_yolov8 .

helm_convert_capi_yolov8:
	RESULTS_DIR=$(RESULTS_DIR) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) kompose -f $(DOCKER_COMPOSE) convert -o $(YOLOV8_PACKAGE_NAME) -c
	@sed -i 's/version: [0-9].[0-9].[0-9]/version: '$(RELEASE_VERSION)'/' ./$(YOLOV8_PACKAGE_NAME)/Chart.yaml
	helm package $(YOLOV8_PACKAGE_NAME)

run_capi_yolov8_ensemble: minikube_build_capi_yolov8_ensemble
	RESULTS_DIR=$(RESULTS_DIR) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) kompose -f $(DOCKER_COMPOSE) convert -o $(YOLOV8_PACKAGE_NAME)/ && \
	kubectl apply -f $(YOLOV8_PACKAGE_NAME)

down_capi_yolov8_ensemble:
	kubectl delete -f $(YOLOV8_PACKAGE_NAME)
	minikube stop

helm_install_capi_yolov8_ensemble:
	helm install -f $(YOLOV8_PACKAGE_NAME)/Chart.yaml dls $(YOLOV8_PACKAGE_NAME)

helm_uninstall_capi_yolov8_ensemble:
	helm uninstall dls
