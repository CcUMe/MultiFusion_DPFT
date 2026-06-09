from dprt.models.language_models.qwen3_vl import build_qwen3_vl_adapter


def build_language_model(name: str, *args, **kwargs):
    normalized = name.lower().replace("-", "_")

    if normalized in {"qwen3_vl_8b", "qwen3_vl_8b_instruct", "qwen3vl8b", "qwen_14b", "qwen_14b_instruct", "qwen14b"}:
        return build_qwen3_vl_adapter(*args, **kwargs)

    raise ValueError(f"Unsupported language model adapter: {name}")
