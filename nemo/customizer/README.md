1. Install volcano: https://volcano.sh/en/docs/installation/#install-with-helm

    https://github.com/volcano-sh/volcano/wiki/FAQ
    ```shell
    oc adm policy add-scc-to-user privileged -z volcano-admission-init -n volcano-system
    oc adm policy add-scc-to-user privileged -z volcano-admission -n volcano-system
    oc adm policy add-scc-to-user privileged -z volcano-controllers -n volcano-system
    oc adm policy add-scc-to-user privileged -z volcano-scheduler -n volcano-system
    ```

2. Install nemo-operator: https://docs.nvidia.com/nim-operator/latest/nemo-prerequisites.html#install-nemo-operator

    ```shell
    noglob helm upgrade --install nemo-operator nemo-operator-${VERSION}.tgz -n nemo --set imagePullSecrets[0].name=ngc-secret --set controllerManager.manager.scheduler=volcano
    ```
   
3. Install MLFlow: https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-microservices/customizer.html#mlflow