# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build build_capi_yolov8_ensemble
.PHONY: minikube_build_capi_yolov8_ensemble run_capi_yolov8_ensemble minikube_show_capi_yolov8_log minikube_pod_log
.PHONY: kube_apply_nfd
.PHONY: down_capi_yolov8_ensemble
.PHONY: helm_convert_capi_yolov8_ensemble helm_install_capi_yolov8_ensemble helm_uninstall_capi_yolov8_ensemble

# initial default values
DOCKER_COMPOSE ?= docker-compose.yml
PIPELINE_COUNT ?= 1
RENDER_MODE ?= 0
RETAIL_USE_CASE_ROOT ?= ../../../
RESULTS_DIR ?= /tmp/results

RELEASE_VERSION=3.2.0
YOLOV8_PACKAGE_NAME=capi-yolov8-ensemble
YOLOV8_POD_NAME=$(shell kubectl get pod | grep "capiyolov8ensemble" | awk '{print $$1}')

build: build_capi_yolov8_ensemble

build_capi_yolov8_ensemble:
	echo "Building CAPI YOLOv8 Ensemble container: HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY}"
	docker build -t yolov8capiensemble:dev -f Dockerfile_yolov8 .

minikube_build_capi_yolov8_ensemble:
# clean up in case there is some left-over due to build error
	rm -rf ./pipelines || true
	rm -rf ./models-downloader || true
	cp -R ../pipelines .
	cp config-yolov8.json ./pipelines/capi_yolov8_ensemble/config-yolov8.json
	cp -R $(RETAIL_USE_CASE_ROOT)models-downloader .
	OVMSConfigFileEnv="../config-yolov8.json" ./models-downloader/downloadOVMSModels.sh
# mkdir -p ./models/yolov8/FP32-INT8/1/ && cp ./yolov8n-int8-416.* ./models/yolov8/FP32-INT8/1/
	minikube start --mount-string="/tmp/results:/tmp/hostpath-provisioner/default/capiyolov8ensemble-claim0" --mount
	minikube image build -t yolov8capiensemble:dev -f Dockerfile_yolov8 .

helm_convert_capi_yolov8_ensemble:
	RESULTS_DIR=$(RESULTS_DIR) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) kompose -f $(DOCKER_COMPOSE) convert -o $(YOLOV8_PACKAGE_NAME) -c
	@sed -i 's/version: [0-9].[0-9].[0-9]/version: '$(RELEASE_VERSION)'/' ./$(YOLOV8_PACKAGE_NAME)/Chart.yaml
	helm package $(YOLOV8_PACKAGE_NAME)

kube_apply_nfd:
	kubectl apply -k 'https://github.com/intel/intel-device-plugins-for-kubernetes/deployments/nfd?ref=v0.30.0'
	kubectl apply -k 'https://github.com/intel/intel-device-plugins-for-kubernetes/deployments/nfd/overlays/node-feature-rules?ref=v0.30.0'
	kubectl apply -k 'https://github.com/intel/intel-device-plugins-for-kubernetes/deployments/gpu_plugin/overlays/nfd_labeled_nodes?ref=v0.30.0'

run_capi_yolov8_ensemble: | minikube_build_capi_yolov8_ensemble kube_apply_nfd
	mkdir -p kubernetes && cp ./$(YOLOV8_PACKAGE_NAME)/templates/capiyolov8ensemble-volumes.yaml ./kubernetes/
	RESULTS_DIR=$(RESULTS_DIR) RETAIL_USE_CASE_ROOT=$(RETAIL_USE_CASE_ROOT) RENDER_MODE=$(RENDER_MODE) kompose -f $(DOCKER_COMPOSE) convert -o kubernetes/ && \
	kubectl apply -f kubernetes

minikube_show_capi_yolov8_log:
	@./minikube_check_result_log.sh

minikube_pod_log:
	@echo "pod name is: $(YOLOV8_POD_NAME)"
	@kubectl logs $(YOLOV8_POD_NAME)

down_capi_yolov8_ensemble:
	kubectl delete -f kubernetes
	minikube stop

helm_install_capi_yolov8_ensemble:
	helm install -f $(YOLOV8_PACKAGE_NAME)/Chart.yaml object-detection-capi-yolov8 $(YOLOV8_PACKAGE_NAME)

helm_uninstall_capi_yolov8_ensemble:
	helm uninstall object-detection-capi-yolov8
