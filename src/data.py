import gzip
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from const import (
    IGNORE_LABEL_TOKEN,
    MISSING_TITLE,
    TOKEN_OFFSET,
    WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
    QueryColumns,
    TestColumns,
    TrainColumns,
)


class CrossEncoderPretrainDataset(IterableDataset):
    """
    Iterable dataset for Baidu-ULTR behavioral-feedback pretraining.
    """

    def __init__(
        self,
        files: List[Path],
        max_sequence_length: int,
        masking_rate: float,
        mask_query: bool,
        mask_doc: bool,
        special_tokens: Dict[str, int],
        segment_types: Dict[str, int],
        ignored_titles: List[bytes],
        rank: int = 0,
        world_size: int = 1,
    ):
        self.files = files
        self.max_sequence_length = max_sequence_length
        self.masking_rate = masking_rate
        self.mask_query = mask_query
        self.mask_doc = mask_doc
        self.special_tokens = special_tokens
        self.segment_types = segment_types
        self.ignored_titles = ignored_titles
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        files = self.get_local_files()
        query_no = 0

        for file in files:
            with gzip.open(file, "rb") as f:
                query = None

                for line in f:
                    columns = line.strip(b"\n").split(b"\t")
                    is_query = len(columns) <= 3

                    if is_query:
                        query_no += 1
                        position = 0
                        query = columns[QueryColumns.QUERY]
                        continue

                    title = columns[TrainColumns.TITLE]
                    abstract = columns[TrainColumns.ABSTRACT]

                    click = float(columns[TrainColumns.CLICK])
                    skip = float(columns[TrainColumns.SKIP])
                    dwell_time = float(
                        columns[TrainColumns.DWELLING_TIME]
                    )

                    dwell_time = np.log1p(dwell_time)

                    if title == MISSING_TITLE:
                        # Missing results still occupy a ranking position.
                        position += 1
                        continue

                    if title == WHAT_OTHER_PEOPLE_SEARCHED_TITLE:
                        # This special result does not occupy a ranking position.
                        continue

                    position += 1

                    query_idx = split_idx(query, TOKEN_OFFSET)
                    title_idx = split_idx(title, TOKEN_OFFSET)
                    abstract_idx = split_idx(abstract, TOKEN_OFFSET)

                    tokens, token_types, attention_mask = format_input(
                        query=query_idx,
                        title=title_idx,
                        abstract=abstract_idx,
                        max_tokens=self.max_sequence_length,
                        special_tokens=self.special_tokens,
                        segment_types=self.segment_types,
                    )

                    masked_tokens, labels = self.mask(
                        tokens,
                        token_types,
                    )

                    yield {
                        "query_id": query_no,
                        "tokens": masked_tokens,
                        "attention_mask": attention_mask,
                        "token_types": token_types,
                        "labels": labels,
                        "clicks": click,
                        "skips": skip,
                        "dwell_time": dwell_time,
                        "positions": position,
                        "partition_name": str(file),
                        "query_no": query_no,
                    }

    def mask(
        self,
        tokens: np.ndarray,
        token_types: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply masked-language-model masking.
        """
        tokens = tokens.copy()

        masking_probability = np.full_like(
            tokens,
            fill_value=self.masking_rate,
            dtype=float,
        )

        mask = np.random.binomial(
            1,
            p=masking_probability,
        ).astype(bool)

        # Never mask special tokens.
        mask[tokens < TOKEN_OFFSET] = False

        if not self.mask_query:
            mask[
                token_types == self.segment_types["QUERY"]
            ] = False

        if not self.mask_doc:
            mask[
                token_types == self.segment_types["TEXT"]
            ] = False

        labels = tokens.copy()
        labels[~mask] = IGNORE_LABEL_TOKEN

        tokens[mask] = self.special_tokens["MASK"]

        return tokens, labels

    def get_local_files(self) -> List[Path]:
        """
        Assign input files across distributed ranks and DataLoader workers.
        """
        rank_files = [
            file
            for index, file in enumerate(self.files)
            if index % self.world_size == self.rank
        ]

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return rank_files

        return [
            file
            for index, file in enumerate(rank_files)
            if index % worker_info.num_workers == worker_info.id
        ]


def split_idx(text: bytes, offset: int) -> List[int]:
    """
    Convert Baidu-ULTR token IDs to model token IDs.
    """
    return [
        int(token) + offset
        for token in text.split(b"\x01")
        if token.strip()
    ]


def format_input(
    query: List[int],
    title: List[int],
    abstract: List[int],
    max_tokens: int,
    special_tokens: Dict[str, int],
    segment_types: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Format a query-document pair as BERT input.

    [CLS] query [SEP] title [SEP] abstract [SEP] [PAD] ...
    """
    cls_token = special_tokens["CLS"]
    sep_token = special_tokens["SEP"]
    pad_token = special_tokens["PAD"]

    query_tokens = [cls_token] + query + [sep_token]
    query_token_types = [
        segment_types["QUERY"]
    ] * len(query_tokens)

    text_tokens = title + [sep_token] + abstract + [sep_token]
    text_token_types = [
        segment_types["TEXT"]
    ] * len(text_tokens)

    tokens = query_tokens + text_tokens
    token_types = query_token_types + text_token_types

    padding = max(max_tokens - len(tokens), 0)

    tokens = (
        tokens[:max_tokens]
        + padding * [pad_token]
    )

    token_types = (
        token_types[:max_tokens]
        + padding * [segment_types["PAD"]]
    )

    tokens = np.asarray(tokens, dtype=int)
    token_types = np.asarray(token_types, dtype=int)
    attention_mask = tokens > pad_token

    return tokens, token_types, attention_mask


def md5_hash(token_ids: List[int]) -> str:
    """
    Compute the token-sequence hash used to match Baidu-ULTR documents.
    """
    text = " ".join(map(str, token_ids)).encode()
    return hashlib.md5(text).hexdigest()


class EvaluationDataset(IterableDataset):
    """
    Dataset for the expert-annotated Baidu-ULTR evaluation set.
    """

    def __init__(
        self,
        files: List[Path],
        max_sequence_length: int,
        special_tokens: Dict[str, int],
        segment_types: Dict[str, int],
        ignored_titles: List[bytes],
    ):
        self.files = files
        self.max_sequence_length = max_sequence_length
        self.special_tokens = special_tokens
        self.segment_types = segment_types
        self.ignored_titles = ignored_titles

    def __iter__(self):
        query_no = 0

        for file in self.files:
            with open(file, "rb") as f:
                prev_query = None

                for line in f:
                    columns = line.strip(b"\n").split(b"\t")

                    current_query = columns[TestColumns.QUERY]
                    is_same_query = prev_query == current_query

                    if not is_same_query:
                        query_no += 1
                        position = 0
                        prev_query = current_query

                    else:
                        title = columns[TestColumns.TITLE]
                        abstract = columns[TestColumns.ABSTRACT]

                        label = int(columns[TestColumns.LABEL])
                        bucket = int(columns[TestColumns.BUCKET])

                        if title == MISSING_TITLE:
                            # Missing results still occupy a ranking position.
                            position += 1
                            continue

                        if title == WHAT_OTHER_PEOPLE_SEARCHED_TITLE:
                            # This special result does not occupy a ranking position.
                            continue

                        position += 1

                        query_idx = split_idx(
                            prev_query,
                            TOKEN_OFFSET,
                        )
                        title_idx = split_idx(
                            title,
                            TOKEN_OFFSET,
                        )
                        abstract_idx = split_idx(
                            abstract,
                            TOKEN_OFFSET,
                        )

                        (
                            tokens,
                            token_types,
                            attention_mask,
                        ) = format_input(
                            query=query_idx,
                            title=title_idx,
                            abstract=abstract_idx,
                            max_tokens=self.max_sequence_length,
                            special_tokens=self.special_tokens,
                            segment_types=self.segment_types,
                        )

                        query_ids = [
                            int(token)
                            for token in prev_query.split(b"\x01")
                            if token
                        ]

                        title_ids = [
                            int(token)
                            for token in title.split(b"\x01")
                            if token
                        ]

                        abstract_ids = [
                            int(token)
                            for token in abstract.split(b"\x01")
                            if token
                        ]

                        query_md5 = md5_hash(query_ids)
                        text_md5 = md5_hash(
                            title_ids + abstract_ids
                        )

                        yield {
                            "query_id": query_no,
                            "tokens": tokens,
                            "attention_mask": attention_mask,
                            "token_types": token_types,
                            "label": label,
                            "bucket": bucket,
                            "query_md5": query_md5,
                            "text_md5": text_md5,
                        }
                        