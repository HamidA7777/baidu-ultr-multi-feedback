import argparse
import pickle
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison


MODEL_FILES = {
    "Click": "per_query_scores_click.pkl",
    "Skip": "per_query_scores_skip.pkl",
    "Dwell-time": "per_query_scores_dwell.pkl",
    "Click + Dwell-time": "per_query_scores_click_dwell.pkl",
    "Skip + Dwell-time": "per_query_scores_skip_dwell.pkl",
    "Click + Skip": "per_query_scores_click_skip.pkl",
    "Click + Skip + Dwell-time": (
        "per_query_scores_click_skip_dwell.pkl"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run pairwise significance tests between "
            "feedback-fusion variants."
        )
    )

    parser.add_argument(
        "--scores-dir",
        type=Path,
        required=True,
        help="Directory containing the per-query score pickle files.",
    )

    parser.add_argument(
        "--metric",
        default="DCG@10",
        help="Ranking metric to compare.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="Family-wise significance level.",
    )

    return parser.parse_args()


def load_scores(path):
    with path.open("rb") as input_file:
        return pickle.load(input_file)


def main():
    args = parse_args()

    model_scores = {}

    for model_name, filename in MODEL_FILES.items():
        path = args.scores_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Score file not found: {path}"
            )

        scores = load_scores(path)

        if args.metric not in scores:
            raise KeyError(
                f"{args.metric} not found in {path}"
            )

        model_scores[model_name] = scores[args.metric]

    rows = []

    for model_name, scores in model_scores.items():
        for score in scores:
            rows.append(
                {
                    "model": model_name,
                    "score": score,
                }
            )

    source = pd.DataFrame(rows)

    comparison = MultiComparison(
        source["score"],
        source["model"],
    )

    table, _, _ = comparison.allpairtest(
        stats.ttest_ind,
        method="bonf",
        alpha=args.alpha,
    )

    print(f"Metric: {args.metric}")
    print(f"Alpha: {args.alpha}")
    print(table)


if __name__ == "__main__":
    main()
