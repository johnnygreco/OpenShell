#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CONFIG_PATH=${1:?gateway config path is required}
FRAGMENT_PATH=${OPENSHELL_GATEWAY_CONFIG_FRAGMENT:-}

if [[ -z "${FRAGMENT_PATH}" ]]; then
  exit 0
fi
if [[ ! -s "${FRAGMENT_PATH}" ]]; then
  echo "ERROR: OPENSHELL_GATEWAY_CONFIG_FRAGMENT is missing or empty: ${FRAGMENT_PATH}" >&2
  exit 2
fi

printf '\n' >>"${CONFIG_PATH}"
cat -- "${FRAGMENT_PATH}" >>"${CONFIG_PATH}"
