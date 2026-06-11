from dprt.models.language_models.qwen3_vl import build_qwen3_vl_adapter
from dprt.models.language_models.qwen_text import build_qwen_text_adapter


def build_language_model(name: str, *args, **kwargs):
    normalized = name.lower().replace("-", "_")

    if normalized in {"qwen3_vl_8b", "qwen3_vl_8b_instruct", "qwen3vl8b"}:
        return build_qwen3_vl_adapter(*args, **kwargs)

    if normalized in {"qwen_7b", "qwen7b", "qwen_7b_base", "qwen2_5_7b", "qwen2_5_7b_instruct", "qwen25_7b", "qwen_25_7b", "qwen_25_7b_instruct", "qwen_14b", "qwen_14b_instruct", "qwen14b"}:
        return build_qwen_text_adapter(*args, **kwargs)

    raise ValueError(f"Unsupported language model adapter: {name}")
