#!/usr/bin/env python
# coding: utf-8

# # Part 1: Preparing Datasets for Fine-tuning and Evaluation

# This notebook showcases transforming a dataset for finetuning and evaluating an LLM for tool calling with NeMo Microservices.

# ## Prerequisites

# ### Deploy NeMo Microservices
# Ensure the NeMo Microservices platform is up and running, including the model downloading step for `meta/llama-3.2-1b-instruct`. Please refer to the [installation guide](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-platform/index.html) for instructions.

# You can verify the `meta/llama-3.1-8b-instruct` is deployed by querying the NIM endpoint. The response should include a model with an `id` of `meta/llama-3.1-8b-instruct`.

# ```bash
# # URL to NeMo deployment management service
# export NEMO_URL="http://nemo.test"
# 
# curl -X GET "$NEMO_URL/v1/models" \
#   -H "Accept: application/json"
# ```

# ### Set up Developer Environment
# Set up your development environment on your machine. The project uses `uv` to manage Python dependencies. From the root of the project, install dependencies and create your virtual environment:

# ```bash
# uv sync --extra dev
# uv pip install -e .
# source .venv/bin/activate
# ```

# ### Build Llama Stack Image
# Build the Llama Stack image using the virtual environment you just created. For local development, set `LLAMA_STACK_DIR` to ensure your local code is use in the image. To use the production version of `llama-stack`, omit `LLAMA_STACK_DIR`.

# ```bash
# LLAMA_STACK_DIR=$(pwd) llama stack build --distro nvidia --image-type venv
# ```

# ## Setup

# First, import the necessary libraries.

# In[ ]:


import os
import json
import random
from pprint import pprint
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import load_dataset


# Set a random seed for reproducibility.

# In[8]:


SEED = 1234

# Limits to at most N tool properties
LIMIT_TOOL_PROPERTIES = 8

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Define the data root directory and create necessary directoryies for storing processed data.

# In[ ]:


# Processed data will be stored here
DATA_ROOT = os.path.join(os.getcwd(), "sample_data")
CUSTOMIZATION_DATA_ROOT = os.path.join(DATA_ROOT, "customization")
VALIDATION_DATA_ROOT = os.path.join(DATA_ROOT, "validation")
EVALUATION_DATA_ROOT = os.path.join(DATA_ROOT, "evaluation")

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CUSTOMIZATION_DATA_ROOT, exist_ok=True)
os.makedirs(VALIDATION_DATA_ROOT, exist_ok=True)
os.makedirs(EVALUATION_DATA_ROOT, exist_ok=True)


# ## Step 1: Download xLAM Data

# This step loads the xLAM dataset from Hugging Face.
# 
# Ensure that you have followed the prerequisites mentioned above, obtained a Hugging Face access token, and configured it in config.py. In addition to getting an access token, you need to apply for access to the xLAM dataset [here](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), which will be approved instantly.

# In[19]:


from config import HF_TOKEN

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_ENDPOINT"] = "https://huggingface.co"


# In[ ]:


# Download from Hugging Face
dataset = load_dataset("Salesforce/xlam-function-calling-60k")

# Inspect a sample
example = dataset['train'][0]
pprint(example)


# For more details on the structure of this data, refer to the [data structure of the xLAM dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k#structure) in the Hugging Face documentation.

# ## Step 2: Prepare Data for Customization

# For Customization, the NeMo Microservices platform leverages the OpenAI data format, comprised of messages and tools:
# - `messages` include the user query, as well as the ground truth `assistant` response to the query. This response contains the function name(s) and associated argument(s) in a `tool_calls` dict
# - `tools` include a list of functions and parameters available to the LLM to choose from, as well as their descriptions.

# The following helper functions convert a single xLAM JSON data point into OpenAI format.

# In[12]:


def normalize_type(param_type: str) -> str:
    """
    Normalize Python type hints and parameter definitions to OpenAI function spec types.

    Args:
        param_type: Type string that could include default values or complex types

    Returns:
        Normalized type string according to OpenAI function spec
    """
    # Remove whitespace
    param_type = param_type.strip()

    # Handle types with default values (e.g. "str, default='London'")
    if "," in param_type and "default" in param_type:
        param_type = param_type.split(",")[0].strip()

    # Handle types with just default values (e.g. "default='London'")
    if param_type.startswith("default="):
        return "string"  # Default to string if only default value is given

    # Remove ", optional" suffix if present
    param_type = param_type.replace(", optional", "").strip()

    # Handle complex types
    if param_type.startswith("Callable"):
        return "string"  # Represent callable as string in JSON schema
    if param_type.startswith("Tuple"):
        return "array"  # Represent tuple as array in JSON schema
    if param_type.startswith("List["):
        return "array"
    if param_type.startswith("Set") or param_type == "set":
        return "array"  # Represent set as array in JSON schema

    # Map common type variations to OpenAI spec types
    type_mapping: Dict[str, str] = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "List": "array",
        "Dict": "object",
        "set": "array",
        "Set": "array"
    }

    if param_type in type_mapping:
        return type_mapping[param_type]
    else:
        print(f"Unknown type: {param_type}")
        return "string"  # Default to string for unknown types


def convert_tools_to_openai_spec(tools: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # If tools is a string, try to parse it as JSON
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as e:
            print(f"Failed to parse tools string as JSON: {e}")
            return []

    # Ensure tools is a list
    if not isinstance(tools, list):
        print(f"Expected tools to be a list, but got {type(tools)}")
        return []

    openai_tools: List[Dict[str, Any]] = []
    for tool in tools:
        # Check if tool is a dictionary
        if not isinstance(tool, dict):
            print(f"Expected tool to be a dictionary, but got {type(tool)}")
            continue

        # Check if 'parameters' is a dictionary
        if not isinstance(tool.get("parameters"), dict):
            print(f"Expected 'parameters' to be a dictionary, but got {type(tool.get('parameters'))} for tool: {tool}")
            continue



        normalized_parameters: Dict[str, Dict[str, Any]] = {}
        for param_name, param_info in tool["parameters"].items():
            if not isinstance(param_info, dict):
                print(
                    f"Expected parameter info to be a dictionary, but got {type(param_info)} for parameter: {param_name}"
                )
                continue

            # Create parameter info without default first
            param_dict = {
                "description": param_info.get("description", ""),
                "type": normalize_type(param_info.get("type", "")),
            }

            # Only add default if it exists, is not None, and is not an empty string
            default_value = param_info.get("default")
            if default_value is not None and default_value != "":
                param_dict["default"] = default_value

            normalized_parameters[param_name] = param_dict

        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {"type": "object", "properties": normalized_parameters},
            },
        }
        openai_tools.append(openai_tool)
    return openai_tools


def save_jsonl(filename, data):
    """Write a list of json objects to a .jsonl file"""
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def convert_tool_calls(xlam_tools):
    """Convert XLAM tool format to OpenAI's tool schema."""
    tools = []
    for tool in json.loads(xlam_tools):
        tools.append({"type": "function", "function": {"name": tool["name"], "arguments": tool.get("arguments", {})}})
    return tools


def convert_example(example, dataset_type='single'):
    """Convert an XLAM dataset example to OpenAI format."""
    obj = {"messages": []}

    # User message
    obj["messages"].append({"role": "user", "content": example["query"]})

    # Tools
    if example.get("tools"):
        obj["tools"] = convert_tools_to_openai_spec(example["tools"])

    # Assistant message
    assistant_message = {"role": "assistant", "content": ""}
    if example.get("answers"):
        tool_calls = convert_tool_calls(example["answers"])

        if dataset_type == "single":
            # Only include examples with a single tool call
            if len(tool_calls) == 1:
                assistant_message["tool_calls"] = tool_calls
            else:
                return None
        else:
            # For other dataset types, include all tool calls
            assistant_message["tool_calls"] = tool_calls

    obj["messages"].append(assistant_message)

    return obj


# The following code cell converts the example data to the OpenAI format required by NeMo Customizer.

# In[13]:


convert_example(example)


# **NOTE**: The convert_example function by default only retains data points that have exactly one tool_call in the output.
# The llama-3.2-1b-instruct model does not support parallel tool calls.
# For more information, refer to the [supported models](https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html#supported-models) in the NeMo documentation.

# ## Process Entire Dataset
# Convert each example by looping through the dataset.

# In[14]:


all_examples = []
with open(os.path.join(DATA_ROOT, "xlam_openai_format.jsonl"), "w") as f:
    for example in dataset["train"]:
        converted = convert_example(example)
        if converted is not None:
            all_examples.append(converted)
            f.write(json.dumps(converted) + "\n")


# ## Split Dataset
# This step splits the dataset into a train, validation, and test set. For demonstration, we use a smaller subset of all the examples.
# You may choose to modify `NUM_EXAMPLES` to leverage a larger subset.

# In[15]:


# Configure to change the size of dataset to use
NUM_EXAMPLES = 5000

assert NUM_EXAMPLES <= len(all_examples), f"{NUM_EXAMPLES} exceeds the total number of available ({len(all_examples)}) data points"


# In[16]:


 # Randomly choose a subset
sampled_examples = random.sample(all_examples, NUM_EXAMPLES)

# Split into 70% training, 15% validation, 15% testing
train_size = int(0.7 * len(sampled_examples))
val_size = int(0.15 * len(sampled_examples))

train_data = sampled_examples[:train_size]
val_data = sampled_examples[train_size : train_size + val_size]
test_data = sampled_examples[train_size + val_size :]

# Save the training and validation splits. We will use test split in the next section
save_jsonl(os.path.join(CUSTOMIZATION_DATA_ROOT, "training.jsonl"), train_data)
save_jsonl(os.path.join(VALIDATION_DATA_ROOT,"validation.jsonl"), val_data)


# ## Step 3: Prepare Data for Evaluation

# For evaluation, the NeMo Microservices platform uses a format with a minor modification to the OpenAI format. This requires `tools_calls` to be brought out of messages to create a distinct parallel field.
# - `messages` includes the user querytools includes a list of functions and parameters available to the LLM to choose from, as well as their descriptions.
# - `tool_calls` is the ground truth response to the user query. This response contains the function name(s) and associated argument(s) in a "tool_calls" dict.

# The following steps transform the test dataset into a format compatible with the NeMo Evaluator microservice.
# This dataset is for measuring accuracy metrics before and after customization.

# In[17]:


def convert_example_eval(entry):
    """Convert a single entry in the dataset to the evaluator format"""

    # Note: This is a WAR for a known bug with tool calling in NIM
    for tool in entry["tools"]:
        if len(tool["function"]["parameters"]["properties"]) > LIMIT_TOOL_PROPERTIES:
            return None

    new_entry = {
        "messages": [],
        "tools": entry["tools"],
        "tool_calls": []
    }

    for msg in entry["messages"]:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            new_entry["tool_calls"] = msg["tool_calls"]
        else:
            new_entry["messages"].append(msg)

    return new_entry

def convert_dataset_eval(data):
    """Convert the entire dataset for evaluation by restructuring the data format."""
    return [result for entry in data if (result := convert_example_eval(entry)) is not None]


# `NOTE`: We have implemented a workaround for a known bug where tool calls freeze the NIM if a tool description includes a function with a larger number of parameters. As such, we have limited the dataset to use examples with available tools having at most 8 parameters. This will be resolved in the next NIM release.

# In[18]:


test_data_eval = convert_dataset_eval(test_data)
save_jsonl(os.path.join(EVALUATION_DATA_ROOT, "xlam-test-single.jsonl"), test_data_eval)

