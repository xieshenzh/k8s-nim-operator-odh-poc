#!/usr/bin/env python
# coding: utf-8

# # Part 3: Model Evaluation Using NeMo Evaluator

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
os.environ["NVIDIA_DATASETS_URL"] = DATA_STORE_URL

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


# In[5]:


from llama_stack.apis.common.job_types import JobStatus

def wait_eval_job(benchmark_id: str, job_id: str, polling_interval: int = 10, timeout: int = 6000):
    start_time = time()
    job_status = client.eval.jobs.status(benchmark_id=benchmark_id, job_id=job_id)

    print(f"Waiting for Evaluation job {job_id} to finish.")
    print(f"Job status: {job_status} after {time() - start_time} seconds.")

    while job_status.status in [JobStatus.scheduled.value, JobStatus.in_progress.value]:
        sleep(polling_interval)
        job_status = client.eval.jobs.status(benchmark_id=benchmark_id, job_id=job_id)

        print(f"Job status: {job_status} after {time() - start_time} seconds.")

        if time() - start_time > timeout:
            raise RuntimeError(f"Evaluation Job {job_id} took more than {timeout} seconds.")

    return job_status


# ## Prerequisites: Configurations and Health Checks
# Before you proceed, make sure that you completed the previous notebooks on data preparation and model fine-tuning to obtain the assets required to follow along.

# ### Configure NeMo Microservices Endpoints
# The following code imports necessary configurations and prints the endpoints for the NeMo Data Store, Entity Store, Customizer, Evaluator, and NIM, as well as the namespace and base model.

# In[ ]:


from config import *

print(f"Data Store endpoint: {DATA_STORE_URL}")
print(f"Entity Store, Customizer, Evaluator endpoint: {NEMO_URL}")
print(f"NIM endpoint: {NIM_URL}")
print(f"Namespace: {NMS_NAMESPACE}")
print(f"Base Model: {BASE_MODEL}")


# ### Check Available Models
# Specify the customized model name that you got from the previous notebook to the following variable.

# In[7]:


# Populate this variable with the value from the previous notebook
# CUSTOMIZED_MODEL = ""
CUSTOMIZED_MODEL = "jgulabrai-1/test-llama-stack@v1"


# The following code verifies that the model has been registed.

# In[ ]:


models = client.models.list()
model_ids = [model.identifier for model in models]

assert CUSTOMIZED_MODEL in model_ids, \
    f"Model {CUSTOMIZED_MODEL} not registered"


# The following code checks if the NIM endpoint hosts the model properly.

# In[12]:


resp = requests.get(f"{NIM_URL}/v1/models")

models = resp.json().get("data", [])
model_names = [model["id"] for model in models]

assert CUSTOMIZED_MODEL in model_names, \
    f"Model {CUSTOMIZED_MODEL} not found"


# ### Verify the Availability of the Datasets
# In the previous notebook, we registered the test dataset along with the train and validation sets.
# The following code performs a sanity check to validate the dataset has been registed with Llama Stack, and exists in NeMo Data Store.

# In[ ]:


repo_id = f"{NMS_NAMESPACE}/{DATASET_NAME}"
print(repo_id)


# In[ ]:


datasets = client.datasets.list()
dataset_ids = [dataset.identifier for dataset in datasets]
assert DATASET_NAME in dataset_ids, \
    f"Dataset {DATASET_NAME} not registered"


# In[ ]:


 # Sanity check to validate dataset
response = requests.get(url=f"{ENTITY_STORE_URL}/v1/datasets/{repo_id}")
assert response.status_code in (200, 201), f"Status Code {response.status_code} Failed to fetch dataset {response.text}"

print("Files URL:", response.json()["files_url"])


# ## Step 1: Establish Baseline Accuracy Benchmark
# First, we’ll assess the accuracy of the 'off-the-shelf' base model—pristine, untouched, and blissfully unaware of the transformative magic that is fine-tuning.
# 

# ### 1.1: Create a Benchmark
# Create a benchmark, which create an evaluation configuration object in NeMo Evaluator. For more information on various parameters, refer to the [NeMo Evaluator configuration](https://developer.nvidia.com/docs/nemo-microservices/evaluate/evaluation-configs.html) in the NeMo microservices documentation.
# - The `tasks.custom-tool-calling.dataset.files_url` is used to indicate which test file to use. Note that it's required to upload this to the NeMo Data Store and register with Entity store before using.
# - The `tasks.dataset.limit` argument below specifies how big a subset of test data to run the evaluation on.
# - The evaluation metric `tasks.metrics.tool-calling-accuracy` reports `function_name_accuracy` and `function_name_and_args_accuracy` numbers, which are as their names imply.

# In[20]:


benchmark_id = "simple-tool-calling-1"
simple_tool_calling_eval_config = {
    "type": "custom",
    "tasks": {
        "custom-tool-calling": {
            "type": "chat-completion",
            "dataset": {
                "files_url": f"hf://datasets/{NMS_NAMESPACE}/{DATASET_NAME}/testing/xlam-test-single.jsonl",
                "limit": 50
            },
            "params": {
                "template": {
                    "messages": "{{ item.messages | tojson}}",
                    "tools": "{{ item.tools | tojson }}",
                    "tool_choice": "auto"
                }
            },
            "metrics": {
                "tool-calling-accuracy": {
                    "type": "tool-calling",
                    "params": {"tool_calls_ground_truth": "{{ item.tool_calls | tojson }}"}
                }
            }
        }
    }
}


# ### 1.2: Register Benchmark
# In order to launch an Evaluation Job using the NeMo Evaluator API, we'll first register a benchmark using the configuration defined in the previous cell.

# In[22]:


response = client.benchmarks.register(
    benchmark_id=benchmark_id,
    dataset_id=repo_id,
    scoring_functions=[],
    metadata=simple_tool_calling_eval_config
)


# ### 1.3: Launch Evaluation Job
# The following code launches an evaluation job. It uses the benchmark defined in the previous cell and targets the base model.

# In[ ]:


# Launch a simple evaluation with the benchmark
response = client.eval.run_eval(
    benchmark_id=benchmark_id,
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": BASE_MODEL,
            "sampling_params": {}
        }
    }
)
job_id = response.model_dump()["job_id"]
print(f"Created evaluation job {job_id}")


# In[ ]:


# Wait for the job to complete
job = wait_eval_job(benchmark_id=benchmark_id, job_id=job_id, polling_interval=5, timeout=600)


# ### 1.4: Review Evaluation Metrics
# The following code gets the evaluation results for the base evaluation job

# In[ ]:


job_results = client.eval.jobs.retrieve(benchmark_id=benchmark_id, job_id=job_id)
print(f"Job results: {json.dumps(job_results.model_dump(), indent=2)}")


# The following code extracts and prints the accuracy scores for the base model.

# In[ ]:


 # Extract function name accuracy score
aggregated_results = job_results.scores[benchmark_id].aggregated_results
base_function_name_accuracy_score = aggregated_results["tasks"]["custom-tool-calling"]["metrics"]["tool-calling-accuracy"]["scores"]["function_name_accuracy"]["value"]
base_function_name_and_args_accuracy = aggregated_results["tasks"]["custom-tool-calling"]["metrics"]["tool-calling-accuracy"]["scores"]["function_name_and_args_accuracy"]["value"]

print(f"Base model: function_name_accuracy: {base_function_name_accuracy_score}")
print(f"Base model: function_name_and_args_accuracy: {base_function_name_and_args_accuracy}")


# ## Step 2: Evaluate the LoRA Customized Model
# 

# ### 2.1 Launch Evaluation Job
# Run another evaluation job with the same benchmark but with the customized model.

# In[ ]:


response = client.eval.run_eval(
    benchmark_id=benchmark_id,
    benchmark_config={
        "eval_candidate": {
            "type": "model",
            "model": CUSTOMIZED_MODEL,
            "sampling_params": {}
        }
    }
)
job_id = response.model_dump()["job_id"]
print(f"Created evaluation job {job_id}")


# In[ ]:


# Wait for the job to complete
job = wait_eval_job(benchmark_id=benchmark_id, job_id=job_id, polling_interval=5, timeout=600)


# ## 2.2 Review Evaluation Metrics

# In[ ]:


job_results = client.eval.jobs.retrieve(benchmark_id=benchmark_id, job_id=job_id)
print(f"Job results: {json.dumps(job_results.model_dump(), indent=2)}")


# In[ ]:


 # Extract function name accuracy score
aggregated_results = job_results.scores[benchmark_id].aggregated_results
ft_function_name_accuracy_score = aggregated_results["tasks"]["custom-tool-calling"]["metrics"]["tool-calling-accuracy"]["scores"]["function_name_accuracy"]["value"]
ft_function_name_and_args_accuracy = aggregated_results["tasks"]["custom-tool-calling"]["metrics"]["tool-calling-accuracy"]["scores"]["function_name_and_args_accuracy"]["value"]

print(f"Custom model: function_name_accuracy: {ft_function_name_accuracy_score}")
print(f"Custom model: function_name_and_args_accuracy: {ft_function_name_and_args_accuracy}")


# A successfully fine-tuned `meta/llama-3.2-1b-instruct` results in a significant increase in tool calling accuracy with.
# 
# In this case you should observe roughly the following improvements -
# - `function_name_accuracy`: 12% to 92%
# - `function_name_and_args_accuracy`: 8% to 72%
# 
# Since this evaluation was on a limited number of samples for demonstration purposes, you may choose to increase `tasks.dataset.limit` in your benchmark `simple_tool_calling_eval_config`.
