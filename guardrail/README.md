1. Create image for the example of connecting to nim
  - Check out branch `nim-test` of https://github.com/xieshenzh/NeMo-Guardrails/tree/nim-test
  - Change the url of `base_url` of nim model `meta/llama3-8b-instruct` in the `config.yml` files for abc_nim and hello_world_nim: examples/bots/abc_nim/config.yml and examples/bots/hello_world_nim/config.yml
  - Build an image and push to image registry: ```podman build -t nemoguardrails --platform=linux/amd64 .```
2. Change the image in the deployment.yaml to be the url of the image built in step 1
3. Deploy guardrail
  - Create namespace `nemo-service`
  - Deploy role.yaml, service_account.yaml, role_binding.yaml, deployment.yaml, service.yaml, route.yaml
  - Once the Deployment is ready, use the url of the Route to access the guardrail service