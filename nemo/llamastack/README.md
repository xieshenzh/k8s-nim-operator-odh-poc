1. Build llamastack distribution-nvidia image: 
```shell
git clone https://github.com/meta-llama/llama-stack.git

cd llama-stack

uv sync --python 3.12
source .venv/bin/activate

uv run --with llama-stack llama stack build --distro nvidia --image-type container

docker tag distribution-nvidia:<tag> quay.io/xiezhang7/distribution-nvidia
docker push quay.io/xiezhang7/distribution-nvidia
```
2. Deploy [deployment](deployment.yaml), [service](service.yaml) and [route](route.yaml) in an OpenShift cluter.
3. Run llamastack nvidia e2e [notebook](https://github.com/meta-llama/llama-stack/tree/main/docs/notebooks/nvidia/beginner_e2e).