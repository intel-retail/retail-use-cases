# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
.PHONY: build-all download-models clean-models build-profile-launcher update-submodules
.PHONY: download-sample-media
.PHONY: run-gst down-gst
.PHONY: clean-results

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
	./models-downloader/downloadOVMSModels.sh

clean-models:
	@find ./models/ -mindepth 1 -maxdepth 1 -type d -exec rm -r {} \;

clean-results:
	@find ./results/ -maxdepth 1 -type f ! -name "result.txt" -exec sudo rm -r {} \;

build-profile-launcher:
	@cd ./core-services && $(MAKE) build-profile-launcher

update-submodules:
	git submodule update --init --recursive

download-sample-media:
	@cd ./performance-tools/benchmark-scripts && ./download_sample_videos.sh

run-gst: download-models download-sample-media
	@cd ./use-cases/dlstreamer && make run

down-gst:
	@cd ./use-cases/dlstreamer && make down
