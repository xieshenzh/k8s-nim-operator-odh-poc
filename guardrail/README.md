1. Deploy `llama3-8b-instruct` NIM 
2. Create image for the example of connecting to nim
  - Check out branch `nim-test` of https://github.com/xieshenzh/NeMo-Guardrails/tree/nim-test
  - Change the url of `base_url` of nim model `meta/llama3-8b-instruct` in the `config.yml` files for abc_nim and hello_world_nim: examples/bots/abc_nim/config.yml and examples/bots/hello_world_nim/config.yml
  - Build an image and push to image registry: ```podman build -t nemoguardrails --platform=linux/amd64 .```
2. Change the image in the `nemoguardrails.yaml` to be the url of the image built in step 1
3. Deploy guardrail
  - Create namespace `nemo-service`
  - Deploy `nemoguardrails.yaml` and `route.yaml`
  - Once the Deployment is ready, use the url of the Route to access the guardrail service
  - Send a request to check if guardrail is working
```shell
curl --insecure <guardrail endpoint url>/v1/chat/completions \                                                                                       ─╯
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "hello_world_nim",    
    "model": "meta/llama3-8b-instruct",
    "messages": [{"role":"user","content":"what is congress?"}],
    "temperature": 0.5,
    "top_p": 1,
    "max_tokens": 1024,
    "stream": false
    }'
```
Expected response:
```json
{
  "messages":[
    {
      "role":"assistant",
      "content":"I'm sorry, I can't respond to that."
    }
  ]
}
```