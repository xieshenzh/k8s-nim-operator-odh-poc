# Install k8s-nim-operator on Openshift
1. Create the Operator namespace:
```oc create namespace nvidia-nim-operator```
2. Add a Docker registry secret that the Operator uses for pulling containers and models from NGC:
```oc create secret -n nvidia-nim-operator docker-registry ngc-secret --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=<ngc-api-key>```
3. Install the Operator:
```operator-sdk run bundle ghcr.io/nvidia/k8s-nim-operator:bundle-latest-release-1.0 --namespace nvidia-nim-operator```

# Deploy the model
1. Create the service namespace:
```oc create namespace nim-service```
2. Add a Docker registry secret that the Operator uses for pulling containers and models from NGC:
```oc create secret -n nim-service docker-registry ngc-secret --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=<ngc-api-key>```
3. Add a generic secret that the container uses to download the model from NVIDIA NGC
```oc create secret -n nim-service generic ngc-api-secret --from-literal=NGC_API_KEY=<ngc-api-key>```
4. Deploy the model
    - Use cache
      - Create NIMCache using ./cache/cache.yaml, and wait for the pod completion
      - Create NIMService using ./cache/service.yaml
    - Use PVC 
      - Create NIMService using ./pvc/service.yaml
