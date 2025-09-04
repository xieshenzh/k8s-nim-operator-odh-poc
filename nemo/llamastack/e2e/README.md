## Get Started

1. Create a virtual environment. This is recommended to isolate project dependencies.

   ```bash
   uv sync --python 3.12
   ```

2. Install the required Python packages.

   ```bash
   uv pip install -e .
   ```

3. Update the following variables in [config.py](./config.py) with your specific URLs and API keys.

   ```python
   # (Required) NeMo Microservices URLs
   NIM_URL = "" # NIM
   DATA_STORE_URL = "" # Data Store
   ENTITY_STORE_URL = "" # Entity Store
   CUSTOMIZER_URL = "" # Customizer
   EVALUATOR_URL = "" # Evaluator 
   GUARDRAIL_URL = "" # Guardrails


   # (Required) Hugging Face Token
   HF_TOKEN = ""
   ```

4. Run the scripts.

   ```bash
   uv run 1_data_preparation.py
   ```

   ```bash
   uv run 2_finetuning_and_inference.py
   ```

   ```bash
   uv run 3_model_evaluation.py   
   ```

   ```bash
   uv run 4_adding_safety_guardrails.py
   ```