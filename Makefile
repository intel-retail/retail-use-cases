# Copyright © 2023 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
.PHONY: build-all download-models build-profile-launcher update-submodules

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
	./models-downloader/downloadModels.sh
	./models-downloader/downloadOVMSModels.sh

build-profile-launcher:
	@cd ./core-services && $(MAKE) build-profile-launcher

upddate-submodules:
	git submodule update --init --recursive