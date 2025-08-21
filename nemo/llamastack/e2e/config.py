# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# (Required) NeMo Microservices URLs
NIM_URL = "" # NIM

DATA_STORE_URL = ""
ENTITY_STORE_URL = ""
CUSTOMIZER_URL = ""
EVALUATOR_URL = ""
GUARDRAIL_URL = ""

# (Required) Configure the base model. Must be one supported by the NeMo Customizer deployment!
BASE_MODEL = "meta/llama-3.2-1b-instruct@v1.0.0+A100"

# (Required) Hugging Face Token
HF_TOKEN = ""

# (Optional) Modify if you've configured a NeMo Data Store token
NDS_TOKEN = "token"

# (Optional) Use a dedicated namespace and dataset name for tutorial assets
NMS_NAMESPACE = "nvidia-tool-calling-tutorial"
DATASET_NAME = "xlam-ft-dataset-1"

# (Optional) Entity Store Project ID. Modify if you've created a project in Entity Store that you'd
# like to associate with your Customized models.
PROJECT_ID = ""
# (Optional) Directory to save the Customized model.
CUSTOMIZED_MODEL_DIR = "nvidia-tool-calling-tutorial/test-llama-stack-4@v1"
