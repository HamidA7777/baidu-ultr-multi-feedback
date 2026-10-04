import argparse
from pathlib import Path

import pandas as pd

from ranking_metrics import REL_METRICS_CUSTOM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ranking scores against expert relevance labels."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="CSV containing query IDs, relevance labels, and ranking scores.",
    )

    parser.add_argument(
        "--score-column",
        required=True,
        help="Column containing the ranking scores to evaluate.",
    )

    return parser.parse_args()


def evaluate_ranking_metrics(df, score_column):
    required_columns = {
        "query_id",
        "label",
        score_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    results = {
        metric: []
        for metric in REL_METRICS_CUSTOM
    }

    for _, group in df.groupby("query_id"):
        labels = group["label"].to_numpy()
        scores = group[score_column].to_numpy()

        for name, metric in REL_METRICS_CUSTOM.items():
            results[name].append(
                float(metric(scores, labels))
            )

    return {
        name: sum(scores) / len(scores)
        for name, scores in results.items()
    }


def main():
    args = parse_args()

    df = pd.read_csv(args.predictions)

    results = evaluate_ranking_metrics(
        df=df,
        score_column=args.score_column,
    )

    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
