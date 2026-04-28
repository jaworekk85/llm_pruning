from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

print("Model loaded")

# Store collected activations here.
activations = []


def hook_fn(module, input, output):
    # Output shape: (batch, seq_len, hidden_dim).
    act = output.detach()
    # Mean activation magnitude.
    mean_act = act.abs().mean().item()
    activations.append(mean_act)


# Attach hooks to MLP layers.
for name, module in model.named_modules():
    if "mlp" in name:
        module.register_forward_hook(hook_fn)

# Test prompts about agriculture.
prompts = [
    "<|user|>\nWhat is agriculture?\n<|assistant|>\n",
    "<|user|>\nExplain crop rotation.\n<|assistant|>\n",
    "<|user|>\nWhat fertilizers are used in farming?\n<|assistant|>\n"
]

print("\nRunning model...\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        model(**inputs)

print("Activations (mean values):\n")

for i, val in enumerate(activations[:20]):  # Only the first 20.
    print(f"Layer {i}: {val:.6f}")
