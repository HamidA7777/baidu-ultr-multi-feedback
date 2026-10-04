import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from const import (
    MASKING_RATE,
    MAX_SEQUENCE_LENGTH,
    MISSING_TITLE,
    SEGMENT_TYPES,
    SPECIAL_TOKENS,
    WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
)
from data import CrossEncoderPretrainDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate behavioral-feedback statistics by rank."
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
        default=Path("utils/ultr_feedback_stats.json"),
        help="Path for the output JSON file.",
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=10_000_000,
        help="Maximum number of documents to process.",
    )

    return parser.parse_args()


def estimate_bias_statistics(dataloader, max_docs=10_000_000):
    k = 10

    impressions = torch.zeros(k + 1, dtype=torch.long)
    clicks = torch.zeros(k + 1, dtype=torch.long)

    examined = torch.zeros(k + 1, dtype=torch.long)
    skips = torch.zeros(k + 1, dtype=torch.long)

    dwell_sum = torch.zeros(k + 1)
    dwell_count = torch.zeros(k + 1)

    doc_counter = 0

    for batch in dataloader:
        positions = batch["positions"]
        click = batch["clicks"]
        skip = batch["skips"]
        dwell_time = batch["dwell_time"]

        for rank in range(1, k + 1):
            mask = positions == rank

            impressions[rank] += mask.sum()

            if click is not None:
                clicks[rank] += click[mask].sum().long()

            if skip is not None:
                skips[rank] += skip[mask].sum().long()
                examined[rank] += mask.sum()

            if dwell_time is not None:
                dwell_sum[rank] += dwell_time[mask].sum()
                dwell_count[rank] += mask.sum()

        doc_counter += len(positions)

        if doc_counter >= max_docs:
            break

    return {
        "docs_processed": doc_counter,
        "clicks": clicks.cpu().tolist(),
        "impressions": impressions.cpu().tolist(),
        "skips": skips.cpu().tolist(),
        "examined": examined.cpu().tolist(),
        "dwell_sum": dwell_sum.cpu().tolist(),
        "dwell_count": dwell_count.cpu().tolist(),
    }


def main():
    args = parse_args()

    batch_size = 64

    files = sorted(args.train_path.glob("*.gz"))

    if not files:
        raise FileNotFoundError(
            f"No .gz files found in {args.train_path}"
        )

    train_dataset = CrossEncoderPretrainDataset(
        files=files,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        masking_rate=MASKING_RATE,
        mask_query=False,
        mask_doc=True,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
        ignored_titles=[
            MISSING_TITLE,
            WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
        ],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    results = estimate_bias_statistics(
        train_dataloader,
        max_docs=args.max_docs,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w") as output_file:
        json.dump(
            results,
            output_file,
            indent=2,
        )

    print(
        f"Processed {results['docs_processed']:,} documents."
    )
    print(f"Saved statistics to {args.output}")


if __name__ == "__main__":
    main()
