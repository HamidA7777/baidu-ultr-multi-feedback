from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import BertConfig

from model import CrossEncoder


HF_REPO_ID = "hamidahmadi/baidu-ultr-multi-feedback-cross-encoders"


# Model variants

MODEL_VARIANTS = {
    "click-only": {
        "checkpoint": "checkpoints/click-only/pretraining/model.pt",
        "multi_task": False,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
    "no-position-debiasing": {
        "checkpoint": "checkpoints/no-position-debiasing/pretraining/model.pt",
        "multi_task": True,
        "debiasing": False,
        "separate_bias_per_task": False,
    },
    "per-task-position-debiasing": {
        "checkpoint": "checkpoints/per-task-position-debiasing/pretraining/model.pt",
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": True,
    },
    "shared-position-debiasing-125k": {
        "checkpoint": (
            "checkpoints/shared-position-debiasing/"
            "pretraining/125k/model.pt"
        ),
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
    "shared-position-debiasing-2m": {
        "checkpoint": (
            "checkpoints/shared-position-debiasing/"
            "pretraining/2m/model.pt"
        ),
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
}


# BERT configuration

def create_bert_config() -> BertConfig:
    """
    Create the BERT configuration used by all released cross-encoders.

    All models use the same BERT architecture and were trained from scratch
    on Baidu-ULTR.
    """

    return BertConfig(
        vocab_size=22_000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    )


# Model loader

def load_model(
    variant: str = "shared-position-debiasing-125k",
    device: Optional[str] = None,
    repo_id: str = HF_REPO_ID,
) -> CrossEncoder:
    """
    Download and load a pretrained Baidu-ULTR cross-encoder.

    Args:
        variant:
            Name of the released model variant.

            Available variants:
                - click-only
                - no-position-debiasing
                - per-task-position-debiasing
                - shared-position-debiasing-125k
                - shared-position-debiasing-2m

        device:
            Device on which to place the model, such as "cpu", "cuda",
            or "cuda:0". If None, CUDA is used when available.

        repo_id:
            Hugging Face repository containing the checkpoints.

    Returns:
        CrossEncoder:
            The requested model in evaluation mode.

    Raises:
        ValueError:
            If the requested model variant is unknown or the checkpoint
            does not contain a model state dictionary.
    """

    if variant not in MODEL_VARIANTS:
        available = "\n".join(
            f"  - {name}" for name in MODEL_VARIANTS
        )

        raise ValueError(
            f"Unknown model variant: '{variant}'.\n\n"
            f"Available variants:\n{available}"
        )

    variant_config = MODEL_VARIANTS[variant]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download checkpoint

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=variant_config["checkpoint"],
    )

    # Instantiate model

    bert_config = create_bert_config()

    model = CrossEncoder(
        config=bert_config,
        multi_task=variant_config["multi_task"],
        debiasing=variant_config["debiasing"],
        separate_bias_per_task=variant_config["separate_bias_per_task"],
    )

    # Load checkpoint

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "model" not in checkpoint:
        raise ValueError(
            f"Checkpoint for '{variant}' does not contain a "
            "'model' state dictionary."
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    # Prepare for inference

    model.to(device)
    model.eval()

    return model


# Available models

def available_models():
    """
    Return the names of all released model variants.
    """

    return list(MODEL_VARIANTS.keys())


if __name__ == "__main__":
    print("Available model variants:")

    for model_name in available_models():
        print(f"  - {model_name}")
