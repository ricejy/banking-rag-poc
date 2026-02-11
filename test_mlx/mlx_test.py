from mlx_lm import load, generate

from mlx_lm import stream_generate

# model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
# response = generate(
#     model, 
#     tokenizer, 
#     prompt="hello! help me generate a sentence on health", 
#     max_tokens=256,
#     verbose=True
# )

model, tokenizer = load(
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    adapter_path="adapters/adapter_iter_one"
)

messages = [{"role": "user", "content": "do not output json. i want to know fund transfer steps."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
# Stream the response token by token
for response in stream_generate(model, tokenizer, prompt, max_tokens=256):
    print(response.text, end="", flush=True)

    

print("=========")
print(response.text)
print("=========")