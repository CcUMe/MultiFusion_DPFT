from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    local_dir="/mnt/disk1/yangqilin/dpft/VLM/QWen/",
    local_dir_use_symlinks=False,
    resume_download=True,
    token="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"
)