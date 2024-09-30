# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build run down model-names

build:
	docker build -t continual_learning:dev .

run:
	docker run -it --rm continual_learning:dev /bin/bash