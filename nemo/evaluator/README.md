1. Install milvus: https://milvus.io/docs/openshift.md
    ```shell
    oc adm policy add-scc-to-user anyuid -z milvus-operator -n milvus-operator
    oc adm policy add-scc-to-user anyuid -z default -n milvus-cluster
    oc adm policy add-scc-to-user anyuid -z cluster-pulsar-zookeeper -n milvus-cluster
    oc adm policy add-scc-to-user anyuid -z cluster-pulsar-bookie -n milvus-cluster
    ```

   Update `milvus-operator-manager-role`
   ```yaml
   - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
      - bind
      - escalate
     apiGroups:
      - rbac.authorization.k8s.io
     resources:
      - clusterrolebindings
      - clusterroles
      - rolebindings
      - roles
   - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
     apiGroups:
      - security.openshift.io
     resources:
      - securitycontextconstraints
   ```
   
2. Install argo-workflows: https://argo-workflows.readthedocs.io/en/latest/installation/#official-release-manifests