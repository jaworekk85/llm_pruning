from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")

# Tokenizer: converts text to token IDs.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Model.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32  # CPU -> float32.
)

print("Model loaded")

# Example question.
prompt = "<|user|>\nExplain agriculture in simple terms.\n<|assistant|>\n"

# Convert text into model input.
inputs = tokenizer(prompt, return_tensors="pt")

# Generate a response.
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# Decode the output tokens back into text.
text = tokenizer.decode(outputs[0])

print("\nRESULT:\n")
print(text)
