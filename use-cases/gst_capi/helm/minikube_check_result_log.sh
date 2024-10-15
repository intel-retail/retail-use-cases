#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

tmpLogFileName=$(minikube ssh ls /tmp/hostpath-provisioner/default/capiyolov8ensemble-claim0/*.log)
logFileName=$(echo $tmpLogFileName | tr -d '\r')
echo ${logFileName}
# show result logs:
minikube ssh cat ${logFileName}
