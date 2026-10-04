from functools import partial

import numpy as np


def dcg_metric(scores, labels, topn):
    """Compute DCG at a given cutoff."""
    sorted_by_scores = np.argsort(scores)[::-1]
    labels = np.take(labels, sorted_by_scores[:topn])

    dcg = 0.0

    for i in range(len(labels)):
        dcg += (2**labels[i] - 1) / np.log2(i + 2)

    return dcg


def ndcg_metric(scores, labels, topn):
    """Compute normalized DCG at a given cutoff."""
    dcg = dcg_metric(scores, labels, topn)

    sorted_labels = np.sort(labels)[::-1]
    idcg = dcg_metric(
        sorted_labels,
        sorted_labels,
        topn,
    )

    return dcg / idcg if idcg > 0.0 else 0.0


def mrr_metric(scores, labels, topn):
    """Compute reciprocal rank of the first relevant document."""
    sorted_by_scores = np.argsort(scores)[::-1]

    for i in range(min(topn, len(sorted_by_scores))):
        if labels[sorted_by_scores[i]] > 0:
            return 1.0 / (i + 1)

    return 0.0


REL_METRICS_CUSTOM = {
    "DCG@1": partial(dcg_metric, topn=1),
    "DCG@3": partial(dcg_metric, topn=3),
    "DCG@5": partial(dcg_metric, topn=5),
    "DCG@10": partial(dcg_metric, topn=10),
    "MRR@10": partial(mrr_metric, topn=10),
    "nDCG@10": partial(ndcg_metric, topn=10),
}
