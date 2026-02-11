import os

from mlx_lm import load, generate

# Load a 4-bit quantized Mistral 7B model 
model, tokenizer = load(
    "mlx-community/gemma-3-12b-it-4bit-DWQ",
    tokenizer_config={"trust_remote_code": True}
)
print(model.model_type)
# Generate text from a prompt
prompt = "explain to me these banking terms: options, short squeeze"
response = generate(
    model, 
    tokenizer, 
    prompt, 
    max_tokens=256,
    verbose=True  # Shows generation stats like speed
)
print(response)