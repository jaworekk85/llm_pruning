"""Shared offline tiny models and local tokenizer; no pretrained downloads."""
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from llm_pruning.models import LoadedModel


@pytest.fixture(params=["llama", "qwen2"])
def tiny_model(request):
    config_cls, model_cls = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
    }[request.param]
    torch.manual_seed(31)
    config = config_cls(
        vocab_size=32, hidden_size=16, intermediate_size=18,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False,
    )
    config._attn_implementation = "eager"
    return model_cls(config).eval()


@pytest.fixture
def tiny_loaded():
    # Byte-level BPE with no merges gives stable prompt boundaries without a download.
    vocabulary = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocabulary.update({char: index + 3 for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    tokenizer_impl = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer_impl.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(19)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=1024, pad_token_id=0, eos_token_id=1,
    )).eval()
    return LoadedModel(model, tokenizer, torch.device("cpu"))
