import argparse
from pathlib import Path

import torch
from transformers import BertConfig

from model import CrossEncoder
from multi_task_fuse import MultiTaskFusionModel


BERT_CONFIG = BertConfig(
    vocab_size=22_000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)

PRETRAINING_CHECKPOINTS = {
    "click-only": {
        "path": "click-only/pretraining/model.pt",
        "multi_task": False,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
    "no-position-debiasing": {
        "path": "no-position-debiasing/pretraining/model.pt",
        "multi_task": True,
        "debiasing": False,
        "separate_bias_per_task": False,
    },
    "per-task-position-debiasing": {
        "path": "per-task-position-debiasing/pretraining/model.pt",
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": True,
    },
    "shared-position-debiasing-125k": {
        "path": "shared-position-debiasing/pretraining/125k/model.pt",
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
    "shared-position-debiasing-2m": {
        "path": "shared-position-debiasing/pretraining/2m/model.pt",
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
}

FUSION_CHECKPOINTS = {
    "click-only": {
        "path": "click-only/fusion",
        "n_features": 1,
    },
    "no-position-debiasing": {
        "path": "no-position-debiasing/fusion",
        "n_features": 3,
    },
    "per-task-position-debiasing": {
        "path": "per-task-position-debiasing/fusion",
        "n_features": 3,
    },
    "shared-position-debiasing-125k": {
        "path": "shared-position-debiasing/fusion/click-skip-dwell",
        "n_features": 3,
    },
    "shared-position-debiasing-2m": {
        "path": "shared-position-debiasing/scaled-2m/fusion",
        "n_features": 3,
    },
    "shared-125k-click": {
        "path": "shared-position-debiasing/fusion/ablations/click",
        "n_features": 1,
    },
    "shared-125k-skip": {
        "path": "shared-position-debiasing/fusion/ablations/skip",
        "n_features": 1,
    },
    "shared-125k-dwell": {
        "path": "shared-position-debiasing/fusion/ablations/dwell",
        "n_features": 1,
    },
    "shared-125k-click-skip": {
        "path": "shared-position-debiasing/fusion/ablations/click-skip",
        "n_features": 2,
    },
    "shared-125k-click-dwell": {
        "path": "shared-position-debiasing/fusion/ablations/click-dwell",
        "n_features": 2,
    },
    "shared-125k-skip-dwell": {
        "path": "shared-position-debiasing/fusion/ablations/skip-dwell",
        "n_features": 2,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check compatibility of released checkpoints."
    )

    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Root directory containing the released checkpoints.",
    )

    return parser.parse_args()


def check_pretraining_checkpoint(name, settings, checkpoint_root):
    path = checkpoint_root / settings["path"]

    if not path.exists():
        print(f"FAIL  pretraining/{name}: not found")
        return False

    model = CrossEncoder(
        config=BERT_CONFIG,
        multi_task=settings["multi_task"],
        debiasing=settings["debiasing"],
        separate_bias_per_task=settings["separate_bias_per_task"],
    )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        if "model" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain a 'model' state dict."
            )

        state_dict = checkpoint["model"]

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        print(f"PASS  pretraining/{name}")
        return True

    except Exception as error:
        print(f"FAIL  pretraining/{name}")
        print(f"      {type(error).__name__}: {error}")
        return False


def check_fusion_checkpoint(
    name,
    path,
    n_features,
):
    if not path.exists():
        print(f"FAIL  fusion/{name}: not found")
        return False

    model = MultiTaskFusionModel(
        n_features=n_features
    )

    try:
        state_dict = torch.load(
            path,
            map_location="cpu",
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        print(
            f"PASS  fusion/{name}/{path.name} "
            f"({n_features} feature"
            f"{'s' if n_features != 1 else ''})"
        )
        return True

    except Exception as error:
        print(f"FAIL  fusion/{name}/{path.name}")
        print(f"      {type(error).__name__}: {error}")
        return False


def main():
    args = parse_args()

    results = {}

    print("Pretraining checkpoints")

    for name, settings in PRETRAINING_CHECKPOINTS.items():
        results[f"pretraining/{name}"] = (
            check_pretraining_checkpoint(
                name=name,
                settings=settings,
                checkpoint_root=args.checkpoint_root,
            )
        )

    print("\nFusion checkpoints")

    for name, settings in FUSION_CHECKPOINTS.items():
        directory = (
            args.checkpoint_root
            / settings["path"]
        )

        for fold in range(1, 4):
            path = directory / f"fold{fold}.pt"
            key = f"fusion/{name}/fold{fold}"

            results[key] = check_fusion_checkpoint(
                name=name,
                path=path,
                n_features=settings["n_features"],
            )

    passed = sum(results.values())
    total = len(results)

    print(f"\n{passed}/{total} checkpoints compatible.")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()