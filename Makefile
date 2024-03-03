# Copyright © 2023 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
.PHONY: build build-cam-sim build-minikube run-minikube-demo stop-minikube-demo

build:
	docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} --target build-default -t dlstreamer:dev -f Dockerfile .

build-cam-sim:
	cp ../../../sample-media/coca-cola-4465029-1920-15-bench.mp4 sample-media/
	docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t cam-sim:dev -f Dockerfile.cam-sim .

build-minikube: build build-cam-sim
	minikube start
	minikube image build -t dlstreamer:dev -f Dockerfile .
	minikube image build -t cam-sim:dev -f Dockerfile.cam-sim .

run-minikube-demo: build-minikube
	cd compose && \
	kompose -f docker-compose.yml convert -o kubernetes/ && \
	kubectl apply -f kubernetes
	
stop-minikube-demo:
	cd compose && \
	kubectl delete -f kubernetes