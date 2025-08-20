#!/usr/bin/env python
# coding: utf-8

# # Part 2: LoRA Fine-tuning Using NeMo Customizer

# In[1]:


import os
import json
import requests
import random
from time import sleep, time
from openai import OpenAI

from config import *


# In[2]:


# Metadata associated with Datasets and Customization Jobs
os.environ["NVIDIA_DATASET_NAMESPACE"] = NMS_NAMESPACE
os.environ["NVIDIA_PROJECT_ID"] = PROJECT_ID

## Inference env vars
os.environ["NVIDIA_BASE_URL"] = NIM_URL

# Data Store env vars
os.environ["NVIDIA_DATASETS_URL"] = ENTITY_STORE_URL

## Customizer env vars
os.environ["NVIDIA_CUSTOMIZER_URL"] = CUSTOMIZER_URL
os.environ["NVIDIA_OUTPUT_MODEL_DIR"] = CUSTOMIZED_MODEL_DIR

# Evaluator env vars
os.environ["NVIDIA_EVALUATOR_URL"] = EVALUATOR_URL

# Guardrails env vars
os.environ["GUARDRAILS_SERVICE_URL"] = GUARDRAIL_URL


# In[ ]:


from llama_stack.core.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient("nvidia")
client.initialize()


# In[15]:


from llama_stack.apis.common.job_types import JobStatus

def wait_customization_job(job_id: str, polling_interval: int = 30, timeout: int = 3600):
    start_time = time()

    res = client.post_training.job.status(job_uuid=job_id)
    job_status = res.status

    print(f"Waiting for Customization job {job_id} to finish.")
    print(f"Job status: {job_status} after {time() - start_time} seconds.")

    while job_status in [JobStatus.scheduled.value, JobStatus.in_progress.value]:
        sleep(polling_interval)
        res = client.post_training.job.status(job_uuid=job_id)
        job_status = res.status

        print(f"Job status: {job_status} after {time() - start_time} seconds.")

        if time() - start_time > timeout:
            raise RuntimeError(f"Customization Job {job_id} took more than {timeout} seconds.")

    return job_status

# When creating a customized model, NIM asynchronously loads the model in its model registry.
# After this, we can run inference with the new model. This helper function waits for NIM to pick up the new model.
def wait_nim_loads_customized_model(model_id: str, polling_interval: int = 10, timeout: int = 300):
    found = False
    start_time = time()

    print(f"Checking if NIM has loaded customized model {model_id}.")

    while not found:
        sleep(polling_interval)

        res = requests.get(f"{NIM_URL}/v1/models")
        if model_id in [model["id"] for model in res.json()["data"]]:
            found = True
            print(f"Model {model_id} available after {time() - start_time} seconds.")
            break
        else:
            print(f"Model {model_id} not available after {time() - start_time} seconds.")

    if not found:
        raise RuntimeError(f"Model {model_id} not available after {timeout} seconds.")

    assert found, f"Could not find model {model_id} in the list of available models."


# ## Prerequisites: Configurations, Health Checks, and Namespaces
# Before you proceed, make sure that you completed the first notebook on data preparation to obtain the assets required to follow along.
# 

# ### Configure NeMo Microservices Endpoints
# This section includes importing required libraries, configuring endpoints, and performing health checks to ensure that the NeMo Data Store, NIM, and other services are running correctly.

# In[ ]:


from config import *

print(f"Data Store endpoint: {DATA_STORE_URL}")
print(f"Entity Store, Customizer, Evaluator endpoint: {NEMO_URL}")
print(f"NIM endpoint: {NIM_URL}")
print(f"Namespace: {NMS_NAMESPACE}")
print(f"Base Model for Customization: {BASE_MODEL}")


# ### Configure Path to Prepared Data
# The following code sets the paths to the prepared dataset files.

# In[6]:


# Path where data preparation notebook saved finetuning and evaluation data
DATA_ROOT = os.path.join(os.getcwd(), "sample_data")
CUSTOMIZATION_DATA_ROOT = os.path.join(DATA_ROOT, "customization")
VALIDATION_DATA_ROOT = os.path.join(DATA_ROOT, "validation")
EVALUATION_DATA_ROOT = os.path.join(DATA_ROOT, "evaluation")

# Sanity checks
train_fp = f"{CUSTOMIZATION_DATA_ROOT}/training.jsonl"
assert os.path.exists(train_fp), f"The training data at '{train_fp}' does not exist. Please ensure that the data was prepared successfully."

val_fp = f"{VALIDATION_DATA_ROOT}/validation.jsonl"
assert os.path.exists(val_fp), f"The validation data at '{val_fp}' does not exist. Please ensure that the data was prepared successfully."

test_fp = f"{EVALUATION_DATA_ROOT}/xlam-test-single.jsonl"
assert os.path.exists(test_fp), f"The test data at '{test_fp}' does not exist. Please ensure that the data was prepared successfully."


# ### Resource Organization Using Namespace
# You can use a [namespace](https://developer.nvidia.com/docs/nemo-microservices/manage-entities/namespaces/index.html) to isolate and organize the artifacts in this tutorial.

# ### Create Namespace
# Both Data Store and Entity Store use namespaces. The following code creates namespaces for the tutorial.

# In[ ]:


def create_namespaces(entity_host, ds_host, namespace):
    # Create namespace in Entity Store
    entity_store_url = f"{entity_host}/v1/namespaces"
    res = requests.post(entity_store_url, json={"id": namespace})
    assert res.status_code in (200, 201, 409, 422), \
        f"Unexpected response from Entity Store during namespace creation: {res.status_code}"
    print(res)

    # Create namespace in Data Store
    nds_url = f"{ds_host}/v1/datastore/namespaces"
    res = requests.post(nds_url, data={"namespace": namespace})
    assert res.status_code in (200, 201, 409, 422), \
        f"Unexpected response from Data Store during namespace creation: {res.status_code}"
    print(res)

create_namespaces(entity_host=ENTITY_STORE_URL, ds_host=DATA_STORE_URL, namespace=NMS_NAMESPACE)


# ### Verify Namespaces
# The following [Data Store API](https://developer.nvidia.com/docs/nemo-microservices/api/datastore.html) and [Entity Store API](https://developer.nvidia.com/docs/nemo-microservices/api/entity-store.html) list the namespace created in the previous cell.

# In[ ]:


# Verify Namespace in Data Store
res = requests.get(f"{DATA_STORE_URL}/v1/datastore/namespaces/{NMS_NAMESPACE}")
print(f"Data Store Status Code: {res.status_code}\nResponse JSON: {json.dumps(res.json(), indent=2)}")

# Verify Namespace in Entity Store
res = requests.get(f"{ENTITY_STORE_URL}/v1/namespaces/{NMS_NAMESPACE}")
print(f"Entity Store Status Code: {res.status_code}\nResponse JSON: {json.dumps(res.json(), indent=2)}")


# ### Step 1: Upload Data to NeMo Data Store
# NeMo Data Store supports data management using the Hugging Face `HfApi` Client.
# **Note that this step does not interact with Hugging Face at all, it just uses the client library to interact with NeMo Data Store.** This is in comparison to the previous notebook, where we used the load_dataset API to download the xLAM dataset from Hugging Face's repository.

# More information can be found in the [documentation](https://developer.nvidia.com/docs/nemo-microservices/manage-entities/tutorials/manage-dataset-files.html#set-up-hugging-face-client).

# #### 1.1 Create Repository
# 

# In[ ]:


repo_id = f"{NMS_NAMESPACE}/{DATASET_NAME}"
print(repo_id)


# In[ ]:


from huggingface_hub import HfApi

hf_api = HfApi(endpoint=f"{DATA_STORE_URL}/v1/hf", token="")

# Create repo
hf_api.create_repo(
    repo_id=repo_id,
    repo_type='dataset',
)


# Next, creating a dataset programmatically requires two steps: uploading and registration. More information can be found in documentation.

# #### 1.2 Upload Dataset Files to NeMo Data Store

# In[ ]:


hf_api.upload_file(path_or_fileobj=train_fp,
    path_in_repo="training/training.jsonl",
    repo_id=repo_id,
    repo_type='dataset',
)

hf_api.upload_file(path_or_fileobj=val_fp,
    path_in_repo="validation/validation.jsonl",
    repo_id=repo_id,
    repo_type='dataset',
)

hf_api.upload_file(path_or_fileobj=test_fp,
    path_in_repo="testing/xlam-test-single.jsonl",
    repo_id=repo_id,
    repo_type='dataset',
)


# Other tips:
# - Take a look at the path_in_repo argument above. If there are more than one files in the subfolders:
#     - All the .jsonl files in training/ will be merged and used for training by customizer.
#     - All the .jsonl files in validation/ will be merged and used for validation by customizer.
# - NeMo Data Store generally supports data management using the [HfApi API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api). For example, to delete a repo, you may use:
#     ```
#     hf_api.delete_repo(
#         repo_id=repo_id,
#         repo_type="dataset"
#     )
#     ```

# #### 1.3 Register the Dataset with NeMo Entity Store
# To use a dataset for operations such as evaluations and customizations, first register the dataset to refer to it by its namespace and name afterward.

# In[ ]:


# client.datasets.register(...)
response = client.datasets.register(
    purpose="post-training/messages",
    dataset_id=DATASET_NAME,
    source={
        "type": "uri",
        "uri": f"hf://datasets/{repo_id}"
    },
    metadata={
        "format": "json",
        "description": "Tool calling xLAM dataset in OpenAI ChatCompletions format",
        "provider": "nvidia"
    }
)
print(response)


# In[ ]:


 # Sanity check to validate dataset
res = requests.get(url=f"{ENTITY_STORE_URL}/v1/datasets/{NMS_NAMESPACE}/{DATASET_NAME}")
assert res.status_code in (200, 201), f"Status Code {res.status_code} Failed to fetch dataset {res.text}"
dataset_obj = res.json()

print("Files URL:", dataset_obj["files_url"])
assert dataset_obj["files_url"] == f"hf://datasets/{repo_id}"


# ### 2. LoRA Customization with NeMo Customizer
# 

# #### 2.1 Start the Training Job
# Start the training job with the Llama Stack Post-Training client.

# In[ ]:


res = client.post_training.supervised_fine_tune(
    job_uuid="",
    model="meta/llama-3.2-1b-instruct@v1.0.0+A100",
    training_config={
        "n_epochs": 2,
        "data_config": {
            "batch_size": 16,
            "dataset_id": DATASET_NAME # NOTE: Namespace is set by `NMS_NAMESPACE` env var
        },
        "optimizer_config": {
            "learning_rate": 0.0001
        }
    },
    algorithm_config={
        "type": "LoRA",
        "adapter_dim": 32,
        "adapter_dropout": 0.1,
         "alpha": 16,
        # NOTE: These fields are required by `AlgorithmConfig` model, but not directly used by NVIDIA
        "rank": 8,
        "lora_attn_modules": [],
        "apply_lora_to_mlp": True,
        "apply_lora_to_output": False
    },
    hyperparam_search_config={},
    logger_config={},
    checkpoint_dir="",
)
print(res)


# In[10]:


job = res.model_dump()

# To job track status
JOB_ID = job["id"]

# This will be the name of the model that will be used to send inference queries to
CUSTOMIZED_MODEL = job["output_model"]


# Tips:
# - To cancel a job that you scheduled incorrectly, run the following code:
# `requests.post(f"{NEMO_URL}/v1/customization/jobs/{JOB_ID}/cancel")`

# #### 2.2 Get Job Status
# The following code polls for the job's status until completion. The training job will take approximately 45 minutes to complete.

# In[ ]:


# Wait for the job to complete
job_status = wait_customization_job(job_id=JOB_ID)


# **IMPORTANT:** Monitor the job status. Ensure training is completed before proceeding by observing the status in the response frame.

# #### 2.3 Validate Availability of Custom Model
# The following NeMo Entity Store API should display the model when the training job is complete. The list below shows all models filtered by your namespace and sorted by the latest first. For more information about this API, see the [NeMo Entity Store API reference](https://developer.nvidia.com/docs/nemo-microservices/api/entity-store.html). With the following code, you can find all customized models, including the one trained in the previous cells.
# Look for the name fields in the output, which should match your `CUSTOMIZED_MODEL`.

# In[ ]:


response = requests.get(f"{ENTITY_STORE_URL}/v1/models", params={"filter[namespace]": NMS_NAMESPACE, "sort" : "-created_at"})

assert response.status_code == 200, f"Status Code {response.status_code}: Request failed. Response: {response.text}"
print("Response JSON:", json.dumps(response.json(), indent=4))


# **Tips:**
# - You can also find the model with its name directly:
#     ```
#     # To specifically get the custom model, you may use the following API -
#     response = requests.get(f"{NEMO_URL}/v1/models/{CUSTOMIZED_MODEL}")
# 
#     assert response.status_code == 200, f"Status Code {response.status_code}: Request failed. Response: {response.text}"
#     print("Response JSON:", json.dumps(response.json(), indent=4))
#     ```

# After the fine-tuning job succeeds, we can't immediately run inference on the customized model. In the background, NIM will load newly-created models and make them available for inference. This process typically takes < 5 minutes - here, we wait for our customized model to be picked up before attempting to run inference.

# In[ ]:


# Check that the customized model has been picked up by NIM;
# We allow up to 5 minutes for the LoRA adapter to be loaded
wait_nim_loads_customized_model(model_id=CUSTOMIZED_MODEL)


# In[17]:


# Check if the custom LoRA model is hosted by NVIDIA NIM
resp = requests.get(f"{NIM_URL}/v1/models")

models = resp.json().get("data", [])
model_names = [model["id"] for model in models]

assert CUSTOMIZED_MODEL in model_names, \
    f"Model {CUSTOMIZED_MODEL} not found"


# #### 2.4 Register Customized Model with Llama Stack
# In order to run inference on the Customized Model with Llama Stack, we need to register the model.

# In[ ]:


from llama_stack.apis.models.models import ModelType

client.models.register(
    model_id=CUSTOMIZED_MODEL,
    model_type=ModelType.llm,
    provider_id="nvidia",
)


# ### Step 3: Sanity Test the Customized Model By Running Sample Inference
# Once the model is customized, its adapter is automatically saved in NeMo Entity Store and is ready to be picked up by NVIDIA NIM.
# You can test the model by making a Chat Completion request. First, choose one of the examples from the test set.

# #### 3.1 Get Test Data Sample

# In[ ]:


def read_jsonl(file_path):
    """Reads a JSON Lines file and yields parsed JSON objects"""
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:
                continue  # Skip empty lines
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue


test_data = list(read_jsonl(test_fp))

print(f"There are {len(test_data)} examples in the test set")


# In[ ]:


 # Randomly choose
test_sample = random.choice(test_data)

# Transform tools to format expected by Llama Stack client
for i, tool in enumerate(test_sample['tools']):
    # Extract properties we will map to the expected format
    tool = tool.get('function', {})
    tool_name = tool.get('name')
    tool_description = tool.get('description')
    tool_params = tool.get('parameters', {})
    tool_params_properties = tool_params.get('properties', {})

    # Create object of parameters for this tool
    transformed_parameters = {}
    for name, property in tool_params_properties.items():
        transformed_param = {
            'param_type': property.get('type'),
            'description': property.get('description')
        }
        if 'default' in property:
            transformed_param['default'] = property['default']
        if 'required' in property:
            transformed_param['required'] = property['required']

        transformed_parameters[name] = transformed_param

    # Update this tool in-place using the expected format
    test_sample['tools'][i] = {
        'tool_name': tool_name,
        'description': tool_description,
        'parameters': transformed_parameters
    }

# Visualize the inputs to the LLM - user query and available tools
test_sample['messages']
test_sample['tools']


# #### 3.2 Send an Inference Call to NIM
# NIM exposes an OpenAI-compatible completions API endpoint, which you can query using Llama Stack inference provider.

# In[ ]:


completion = client.inference.chat_completion(
    model_id=CUSTOMIZED_MODEL,
    messages=test_sample["messages"],
    tools=test_sample["tools"],
    tool_choice="auto",
    stream=False,
    sampling_params={
        "max_tokens": 512,
        "strategy": {
            "type": "top_p",
            "temperature": 0.1,
            "top_p": 0.7,
        }
    },
)

completion.completion_message.tool_calls


# Given that the fine-tuning job was successful, you can get an inference result comparable to the ground truth:

# In[ ]:


# The ground truth answer
test_sample['tool_calls']


# #### 3.3 Take Note of Your Custom Model Name
# Take note of your custom model name, as you will use it to run evaluations in the subsequent notebook.

# In[ ]:


print(f"Name of your custom model is: {CUSTOMIZED_MODEL}")

