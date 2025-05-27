k8s-nim-operator guide (version 2.0): https://docs.nvidia.com/nim-operator/latest/nemo-prerequisites.html

Nemo microservices guide (version 25.4.0): https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-microservices/index.html

Jupyter notebook PoC: https://github.com/NVIDIA/k8s-nim-operator/blob/main/test/e2e/jupyter-notebook/e2e-notebook.ipynb

1. Create a namespace: https://docs.nvidia.com/nim-operator/latest/nemo-prerequisites.html#create-nemo-namespace
2. Create Secrets in the namespace:
    ```shell
    create secret -n nemo docker-registry ngc-secret \
      --docker-server=nvcr.io \
      --docker-username='$oauthtoken' \
      --docker-password=<ngc-api-key>
    ```
    ```shell
    oc create secret -n nemo generic ngc-api-secret \
      --from-literal=NGC_API_KEY=<ngc-api-key>
    ```