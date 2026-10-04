import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from const import (
    MAX_SEQUENCE_LENGTH,
    MISSING_TITLE,
    SEGMENT_TYPES,
    SPECIAL_TOKENS,
    WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
)
from data import EvaluationDataset
from load_model import MODEL_VARIANTS, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate predictions on the Baidu-ULTR "
            "expert-annotated evaluation set."
        )
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        required=True,
        help="Evaluation file or directory containing evaluation .txt files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output prediction CSV.",
    )

    parser.add_argument(
        "--variant",
        choices=MODEL_VARIANTS.keys(),
        default="shared-position-debiasing-125k",
        help="Released model variant to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, for example cuda or cpu.",
    )

    return parser.parse_args()


def evaluate(model, dataloader, save_path, device):
    model.to(device)
    model.eval()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model.multi_task:
        fieldnames = [
            "query_id",
            "query_md5",
            "text_md5",
            "label",
            "bucket",
            "click_scores",
            "skip_scores",
            "dwell_time_scores",
        ]
    else:
        fieldnames = [
            "query_id",
            "query_md5",
            "text_md5",
            "label",
            "bucket",
            "click_scores",
        ]

    with save_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for batch in dataloader:
            batch["tokens"] = batch["tokens"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["token_types"] = batch["token_types"].to(device)

            if model.multi_task:
                (
                    click_scores,
                    skip_scores,
                    dwell_time_scores,
                ) = model.predict(
                    tokens=batch["tokens"],
                    attention_mask=batch["attention_mask"],
                    token_types=batch["token_types"],
                )

                click_scores = click_scores.detach().cpu()
                skip_scores = skip_scores.detach().cpu()
                dwell_time_scores = dwell_time_scores.detach().cpu()

                for i in range(len(click_scores)):
                    writer.writerow(
                        {
                            "query_id": batch["query_id"][i].item(),
                            "query_md5": batch["query_md5"][i],
                            "text_md5": batch["text_md5"][i],
                            "label": batch["label"][i].item(),
                            "bucket": batch["bucket"][i].item(),
                            "click_scores": click_scores[i].item(),
                            "skip_scores": skip_scores[i].item(),
                            "dwell_time_scores": (
                                dwell_time_scores[i].item()
                            ),
                        }
                    )

            else:
                click_scores = model.predict(
                    tokens=batch["tokens"],
                    attention_mask=batch["attention_mask"],
                    token_types=batch["token_types"],
                )

                click_scores = click_scores.detach().cpu()

                for i in range(len(click_scores)):
                    writer.writerow(
                        {
                            "query_id": batch["query_id"][i].item(),
                            "query_md5": batch["query_md5"][i],
                            "text_md5": batch["text_md5"][i],
                            "label": batch["label"][i].item(),
                            "bucket": batch["bucket"][i].item(),
                            "click_scores": click_scores[i].item(),
                        }
                    )


def get_evaluation_files(test_path):
    if test_path.is_file():
        return [test_path]

    files = sorted(test_path.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {test_path}"
        )

    return files


def main():
    args = parse_args()

    device = args.device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    files = get_evaluation_files(args.test_path)

    test_dataset = EvaluationDataset(
        files=files,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
        ignored_titles=[
            MISSING_TITLE,
            WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
        ],
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
    )

    model = load_model(
        variant=args.variant,
        device=device,
    )

    evaluate(
        model=model,
        dataloader=test_dataloader,
        save_path=args.output,
        device=device,
    )


if __name__ == "__main__":
    main()
