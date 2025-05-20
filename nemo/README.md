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