mkdir weights
wget -P weights/ https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/resolve/main/model-00002-of-00002.safetensors?download=true
wget -P weights/ https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/resolve/main/model-00001-of-00002.safetensors?download=true
wget -P weights/ https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/raw/main/model.safetensors.index.json

mkdir tokenizer
wget -P tokenizer/ https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/raw/main/tokenizer.json
wget -P tokenizer/ https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/raw/main/tokenizer_config.json
