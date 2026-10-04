import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from ranking_metrics import REL_METRICS_CUSTOM


TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-3
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
KFOLDS = 3
VAL_SPLIT = 0.1

FEATURE_MAP = {
    "click": ["click_scores"],
    "skip": ["skip_scores"],
    "dwell": ["dwell_time_scores"],
    "click-skip": ["click_scores", "skip_scores"],
    "click-dwell": ["click_scores", "dwell_time_scores"],
    "skip-dwell": ["skip_scores", "dwell_time_scores"],
    "click-skip-dwell": [
        "click_scores",
        "skip_scores",
        "dwell_time_scores",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the behavioral-feedback fusion model."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Pre-fusion prediction CSV.",
    )

    parser.add_argument(
        "--bm25",
        type=Path,
        required=True,
        help="BM25 evaluation feather file.",
    )

    parser.add_argument(
        "--features",
        choices=FEATURE_MAP.keys(),
        default="click-skip-dwell",
        help="Behavioral signals to fuse.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for fold checkpoints and per-query scores.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, for example cuda or cpu.",
    )

    return parser.parse_args()


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiTaskFusionDataset(Dataset):
    def __init__(self, df, features, label):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        features = torch.tensor(
            row[self.features].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        label = torch.tensor(
            row[self.label],
            dtype=torch.float32,
        )

        return {
            "query_id": row["query_id"],
            "label": label,
            "bucket": row["bucket"],
            "features": features,
        }


class MultiTaskFusionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.n_features = n_features

        self.multi_task_fusion_model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.multi_task_fusion_model(x)


def evaluate_ranking_metrics(
    model,
    dataloader,
    results_dict,
    per_query_dict,
    device,
):
    model.to(device)
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["features"].to(device)
            fused_scores = model(inputs).squeeze(-1).cpu()

            for i in range(len(fused_scores)):
                rows.append(
                    {
                        "query_id": batch["query_id"][i].item(),
                        "label": batch["label"][i].item(),
                        "bucket": batch["bucket"][i].item(),
                        "fused_scores": fused_scores[i].item(),
                    }
                )

    df = pd.DataFrame(rows)

    fold_results = {
        metric: []
        for metric in REL_METRICS_CUSTOM
    }

    for _, group in df.groupby("query_id"):
        sorted_group = group.sort_values(
            by="fused_scores",
            ascending=False,
        )

        labels = sorted_group["label"].values
        scores = sorted_group["fused_scores"].values

        for name, metric_function in REL_METRICS_CUSTOM.items():
            score = float(
                metric_function(scores, labels)
            )
            fold_results[name].append(score)

    for name, scores in fold_results.items():
        results_dict[name].append(np.mean(scores))
        per_query_dict[name].extend(scores)


def evaluate_bm25(df, results_dict):
    fold_results = {
        metric: []
        for metric in REL_METRICS_CUSTOM
    }

    for _, group in df.groupby("query_id"):
        sorted_group = group.sort_values(
            by="bm25",
            ascending=False,
        )

        labels = sorted_group["label"].values
        scores = sorted_group["bm25"].values

        for name, metric_function in REL_METRICS_CUSTOM.items():
            score = float(
                metric_function(scores, labels)
            )
            fold_results[name].append(score)

    for name, scores in fold_results.items():
        results_dict[name].append(np.mean(scores))


def train_fusion_kfold(
    dataset,
    df,
    features,
    output_dir,
    device,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()

    query_df = (
        df.groupby("query_id")
        .agg({"label": "mean"})
        .reset_index()
    )

    query_counts = df["query_id"].value_counts()

    query_df["query_freq"] = (
        query_df["query_id"].map(query_counts)
    )

    query_df["stratify_bin"] = pd.qcut(
        query_df["query_freq"],
        q=KFOLDS,
        labels=False,
        duplicates="drop",
    )

    skf = StratifiedKFold(
        n_splits=KFOLDS,
        shuffle=True,
        random_state=1,
    )

    query_ids = query_df["query_id"].values
    stratify_bins = query_df["stratify_bin"].values

    results_per_fold = {
        metric: []
        for metric in REL_METRICS_CUSTOM
    }

    results_per_fold_bm25 = {
        metric: []
        for metric in REL_METRICS_CUSTOM
    }

    per_query_scores = {
        metric: []
        for metric in REL_METRICS_CUSTOM
    }

    df_reset = df.reset_index(drop=True)

    splits = skf.split(
        query_ids,
        stratify_bins,
    )

    for fold, (train_val_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        print(f"\nFold {fold}/{KFOLDS}")

        model = MultiTaskFusionModel(
            n_features=len(features)
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        train_val_query_ids = query_ids[train_val_idx]

        # 10% of all queries are used for validation.
        n_val = int(VAL_SPLIT * len(query_ids))

        val_query_ids = set(
            train_val_query_ids[:n_val]
        )
        train_query_ids = set(
            train_val_query_ids[n_val:]
        )
        test_query_ids = set(
            query_ids[test_idx]
        )

        print(
            f"Train queries: {len(train_query_ids)}, "
            f"val queries: {len(val_query_ids)}, "
            f"test queries: {len(test_query_ids)}"
        )

        train_row_idx = df_reset[
            df_reset["query_id"].isin(train_query_ids)
        ].index.tolist()

        val_row_idx = df_reset[
            df_reset["query_id"].isin(val_query_ids)
        ].index.tolist()

        test_row_idx = df_reset[
            df_reset["query_id"].isin(test_query_ids)
        ].index.tolist()

        train_loader = DataLoader(
            Subset(dataset, train_row_idx),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )

        val_loader = DataLoader(
            Subset(dataset, val_row_idx),
            batch_size=TEST_BATCH_SIZE,
        )

        test_loader = DataLoader(
            Subset(dataset, test_row_idx),
            batch_size=TEST_BATCH_SIZE,
        )

        test_df = df_reset[
            df_reset["query_id"].isin(test_query_ids)
        ]

        fold_save_path = output_dir / f"fold{fold}.pt"

        fold_best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                inputs = batch["features"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                output = model(inputs).squeeze(-1)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = (
                total_train_loss / len(train_loader)
            )

            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["features"].to(device)
                    labels = batch["label"].to(device)

                    output = model(inputs).squeeze(-1)

                    total_val_loss += criterion(
                        output,
                        labels,
                    ).item()

            avg_val_loss = (
                total_val_loss / len(val_loader)
            )

            print(
                f"epoch={epoch + 1} "
                f"train_loss={avg_train_loss:.6f} "
                f"val_loss={avg_val_loss:.6f}"
            )

            if avg_val_loss < fold_best_loss:
                fold_best_loss = avg_val_loss
                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    fold_save_path,
                )
            else:
                epochs_without_improvement += 1

                if (
                    epochs_without_improvement
                    >= EARLY_STOPPING_PATIENCE
                ):
                    print(
                        f"Early stopping at epoch {epoch + 1}"
                    )
                    break

        model.load_state_dict(
            torch.load(
                fold_save_path,
                map_location=device,
            )
        )

        evaluate_ranking_metrics(
            model=model,
            dataloader=test_loader,
            results_dict=results_per_fold,
            per_query_dict=per_query_scores,
            device=device,
        )

        evaluate_bm25(
            df=test_df,
            results_dict=results_per_fold_bm25,
        )

        model_results = {
            name: round(values[-1], 4)
            for name, values in results_per_fold.items()
        }

        bm25_results = {
            name: round(values[-1], 4)
            for name, values in results_per_fold_bm25.items()
        }

        print(f"Model: {model_results}")
        print(f"BM25: {bm25_results}")

    results_mean = {
        metric: np.mean(scores)
        for metric, scores in results_per_fold.items()
    }

    results_std = {
        metric: np.std(scores)
        for metric, scores in results_per_fold.items()
    }

    results_mean_bm25 = {
        metric: np.mean(scores)
        for metric, scores in results_per_fold_bm25.items()
    }

    results_std_bm25 = {
        metric: np.std(scores)
        for metric, scores in results_per_fold_bm25.items()
    }

    print("\nCV results")

    for metric in results_per_fold:
        print(
            f"{metric}: "
            f"{results_mean[metric]:.4f} "
            f"± {results_std[metric]:.4f}"
        )

        print(
            f"BM25, {metric}: "
            f"{results_mean_bm25[metric]:.4f} "
            f"± {results_std_bm25[metric]:.4f}"
        )

    return (
        results_mean,
        results_std,
        results_mean_bm25,
        results_std_bm25,
        per_query_scores,
    )


def main():
    args = parse_args()
    set_seed()

    device = args.device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    features = FEATURE_MAP[args.features]

    evaluation_df = pd.read_csv(args.predictions)
    bm25_df = pd.read_feather(args.bm25)

    missing_features = [
        feature
        for feature in features
        if feature not in evaluation_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Prediction file is missing columns: "
            + ", ".join(missing_features)
        )

    evaluation_df = evaluation_df.merge(
        bm25_df[
            ["query_md5", "text_md5", "bm25"]
        ],
        on=["query_md5", "text_md5"],
        how="left",
    )

    evaluation_df["bm25"] = (
        evaluation_df["bm25"].fillna(0.0)
    )

    dataset = MultiTaskFusionDataset(
        df=evaluation_df,
        features=features,
        label="label",
    )

    (
        _,
        _,
        _,
        _,
        per_query_scores,
    ) = train_fusion_kfold(
        dataset=dataset,
        df=evaluation_df,
        features=features,
        output_dir=args.output_dir,
        device=device,
    )

    scores_path = (
        args.output_dir
        / f"per_query_scores_{args.features.replace('-', '_')}.pkl"
    )

    with scores_path.open("wb") as output_file:
        pickle.dump(
            per_query_scores,
            output_file,
        )


if __name__ == "__main__":
    main()
