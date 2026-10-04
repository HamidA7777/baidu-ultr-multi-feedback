import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from const import (
    MAX_SEQUENCE_LENGTH,
    SEGMENT_TYPES,
    SPECIAL_TOKENS,
    TOKEN_OFFSET,
)
from data import format_input, split_idx
from load_model import load_model


def prepare_input(
    query: bytes,
    title: bytes,
    abstract: bytes,
    device: str,
):
    """
    Prepare a Baidu-ULTR query-document pair for the cross-encoder.

    The inputs are byte strings containing Baidu token IDs separated
    by the \\x01 delimiter.
    """

    query_ids = split_idx(query, TOKEN_OFFSET)
    title_ids = split_idx(title, TOKEN_OFFSET)
    abstract_ids = split_idx(abstract, TOKEN_OFFSET)

    tokens, token_types, attention_mask = format_input(
        query=query_ids,
        title=title_ids,
        abstract=abstract_ids,
        max_tokens=MAX_SEQUENCE_LENGTH,
        special_tokens=SPECIAL_TOKENS,
        segment_types=SEGMENT_TYPES,
    )

    tokens = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    attention_mask = torch.tensor(
        attention_mask,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    token_types = torch.tensor(
        token_types,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    return tokens, attention_mask, token_types


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        variant="shared-position-debiasing-125k",
        device=device,
    )

    # Example in the native Baidu-ULTR token-ID format.
    #
    # Baidu token IDs are separated by the \\x01 byte delimiter.
    # Replace these example IDs with a query, title, and abstract
    # from Baidu-ULTR.
    query = b"123\x01456\x01789"
    title = b"234\x01567\x01890"
    abstract = b"345\x01678\x01901\x011234"

    tokens, attention_mask, token_types = prepare_input(
        query=query,
        title=title,
        abstract=abstract,
        device=device,
    )

    click_scores, skip_scores, dwell_time_scores = model.predict(
        tokens=tokens,
        attention_mask=attention_mask,
        token_types=token_types,
    )

    print(f"Click score:      {click_scores[0]:.6f}")
    print(f"Skip score:       {skip_scores[0]:.6f}")
    print(f"Dwell-time score: {dwell_time_scores[0]:.6f}")


if __name__ == "__main__":
    main()
