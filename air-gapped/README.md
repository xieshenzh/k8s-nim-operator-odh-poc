## Use proxy
1. [Set](https://docs.nvidia.com/nim/large-language-models/latest/deploy-behind-proxy.html) environment variables in the containers
   1. Environment variables: HTTPS_PROXY(https_proxy), HTTP_PROXY(http_proxy), NO_PROXY(no_proxy)
   2. Set the environment variables via ServingRuntime CRs (or InferenceService CRs if different proxies are used for different models)
2. [Inject](https://docs.nvidia.com/nim-operator/latest/air-gap.html#proxy-support) certificate in the containers
   1. Refer to OpenShift docs on injecting certificate in the containers
3. Set proxy for pulling NIM images
   1. Refer to OpenShift docs on configuring proxy for image registry

## Deploy with pre-downloaded files

### Download model files

1. Download as [model store](https://docs.nvidia.com/nim/large-language-models/latest/deploy-air-gap.html#create-the-model-store)
   1. In a connected environment, [start](./mulit-llm/model-store.yaml) a container with the NIM image of the model to be deployed
   2. Mount the external storage to a directory as the model store
   3. Execute command `create-model-store` and set the model repo and model store to start downloading the files
2. Download as [model cache](https://docs.nvidia.com/nim/large-language-models/latest/deploy-air-gap.html#offline-cache-option)
   1. In a connected environment, [start](./llm/profile.yaml) a container with the NIM image of the model to be deployed
   2. Mount the external storage to the directory `/opt/nim/.cache` as the model cache
   3. Execute command `download-to-cache` to create a cache of the model files (optionally, set a profile of the model if not using the auto-detected profile)
3. Multi-LLM
   1. Multi-LLM models files can be downloaded directly from huggingface
   2. Multi-LLM models files can also be downloaded as model store

### Prepare the environment
1. Pull the NIM images from NGC registry, then push images to the air-gapped cluster
2. Copy the model store or model cache to the air-gapped cluster, then create PVCs with the model store or model cache 

### Deploy models
1. Deploy with model store
   1. Deploy a [ServingRuntime](./mulit-llm/runtime.yaml) CR and an [InferenceService](./mulit-llm/service.yaml) CR.
   2. In the CRs, mount the model store PVC to a directory in the container.
   3. Set environment variable `NIM_MODEL_NAME` to the path of model store directory.
   4. Set environment variable `NIM_SERVED_MODEL_NAME` to the name of the model.
2. Deploy with model cache
    1. Deploy a [ServingRuntime](./llm/runtime.yaml) CR and an [InferenceService](./llm/service.yaml) CR.
    2. In the CRs, mount the model cache PVC to a directory in the container (the default cache directory is `/opt/nim/.cache`).
    3. If not using the default cache directory, set environment variable `NIM_CACHE_PATH` to the path of model cache directory.
    4. If a profile is selected when creating the model cache, set environment variable `NIM_MODEL_PROFILE` to the profile id. 