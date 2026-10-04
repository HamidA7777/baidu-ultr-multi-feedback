import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import BertConfig


VARIANTS = {
    "click-only": {
        "multi_task": False,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
    "no-position-debiasing": {
        "multi_task": True,
        "debiasing": False,
        "separate_bias_per_task": False,
    },
    "per-task-position-debiasing": {
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": True,
    },
    "shared-position-debiasing": {
        "multi_task": True,
        "debiasing": True,
        "separate_bias_per_task": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Baidu-ULTR cross-encoder."
    )

    parser.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Directory containing the Baidu-ULTR pretraining .gz files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the training checkpoint.",
    )

    parser.add_argument(
        "--variant",
        choices=VARIANTS.keys(),
        default="shared-position-debiasing",
        help="Training configuration.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=125_000,
        help="Maximum number of gradient updates.",
    )

    return parser.parse_args()


def set_seed(seed: int = 1):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, path):
    """
    Save model and optimizer state.

    The checkpoint format is kept compatible with the checkpoints
    produced by the original training implementation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    return torch.load(
        path,
        map_location=map_location,
    )


def ddp_setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def train(
    model,
    dataloader,
    optimizer,
    rank,
    local_rank,
    save_path,
    max_steps=125_000,
    save_every=1_000,
):
    """Train a Baidu-ULTR cross-encoder using DistributedDataParallel."""
    device = torch.device(f"cuda:{local_rank}")

    model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )

    model.train()

    total_loss = 0.0
    num_batches = 0
    steps = 0

    for batch in dataloader:
        batch["tokens"] = batch["tokens"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["token_types"] = batch["token_types"].to(device)
        batch["labels"] = batch["labels"].to(device)

        batch["clicks"] = (
            batch["clicks"].to(device).float()
            if batch["clicks"] is not None
            else None
        )

        batch["skips"] = (
            batch["skips"].to(device).float()
            if batch["skips"] is not None
            else None
        )

        batch["dwell_time"] = (
            batch["dwell_time"].to(device).float()
            if batch["dwell_time"] is not None
            else None
        )

        batch["positions"] = batch["positions"].to(device).long()

        optimizer.zero_grad()

        loss_per_task, _ = model(**batch)

        click_loss = loss_per_task.get("click", 0.0)
        skip_loss = loss_per_task.get("skip", 0.0)
        dwell_time_loss = loss_per_task.get("dwell_time", 0.0)
        mlm_loss = loss_per_task.get("mlm", 0.0)

        total_loss_step = (
            click_loss
            + skip_loss
            + dwell_time_loss
            + mlm_loss
        )

        total_loss_step.backward()
        optimizer.step()

        steps += 1
        num_batches += 1
        total_loss += total_loss_step.item()

        if rank == 0 and steps % 100 == 0:
            avg_loss = total_loss / num_batches

            print(
                f"step={steps:,}/{max_steps:,} "
                f"loss={avg_loss:.6f}"
            )

        if rank == 0 and steps % save_every == 0:
            save_checkpoint(
                model.module,
                optimizer,
                save_path,
            )

        if steps >= max_steps:
            break

    if rank == 0:
        save_checkpoint(
            model.module,
            optimizer,
            save_path,
        )


def main():
    args = parse_args()

    rank, local_rank, world_size = ddp_setup()
    set_seed(1 + rank)

    from const import (
        MASKING_RATE,
        MAX_SEQUENCE_LENGTH,
        MISSING_TITLE,
        SEGMENT_TYPES,
        SPECIAL_TOKENS,
        WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
    )
    from data import CrossEncoderPretrainDataset
    from model import CrossEncoder

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    SAVE_EVERY = 1_000

    # Model
    DWELL_LOSS_WEIGHT = 0.1

    # BERT
    VOCAB_SIZE = 22_000
    HIDDEN_SIZE = 768
    NUM_HIDDEN_LAYERS = 12
    NUM_ATTENTION_HEADS = 12
    POSITION_VOCAB_SIZE = 30

    variant = VARIANTS[args.variant]

    files = sorted(args.train_path.glob("*.gz"))

    if not files:
        raise FileNotFoundError(
            f"No .gz files found in {args.train_path}"
        )

    ignored_titles = [
        MISSING_TITLE,
        WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
    ]

    train_dataset = CrossEncoderPretrainDataset(
        files=files,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        masking_rate=MASKING_RATE,
        mask_query=False,
        mask_doc=True,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
        ignored_titles=ignored_titles,
        rank=rank,
        world_size=world_size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
    )

    model = CrossEncoder(
        config=config,
        multi_task=variant["multi_task"],
        debiasing=variant["debiasing"],
        separate_bias_per_task=variant["separate_bias_per_task"],
        dwell_loss_weight=DWELL_LOSS_WEIGHT,
        position_vocab_size=POSITION_VOCAB_SIZE,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if rank == 0:
        print(f"variant={args.variant}")
        print(f"max_steps={args.max_steps:,}")
        print(f"training_files={len(files)}")

    train(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        rank=rank,
        local_rank=local_rank,
        save_path=args.output,
        max_steps=args.max_steps,
        save_every=SAVE_EVERY,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
