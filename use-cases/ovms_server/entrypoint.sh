#!/usr/bin/env bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

if [ "$TARGET_DEVICE" ] 
then
    ovms_jsonCfg=`cat /config/config.json`
    ovms_jsonCfg=`jq --arg device $TARGET_DEVICE '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
    echo $ovms_jsonCfg > /config/config.json
fi

PORT="${PORT:=8090}"
/ovms/bin/ovms --config_path /config/config.json --port $PORT