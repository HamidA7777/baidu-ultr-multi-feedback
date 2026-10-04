from typing import Optional

import torch
from torch import LongTensor, FloatTensor, BoolTensor, IntTensor
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, BertForPreTraining
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class BertModel(BertForPreTraining):
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

    def __init__(self, config: PretrainedConfig):
        super(BertModel, self).__init__(config)

        self.mlm_head = BertLMPredictionHead(config)
        self.mlm_loss = CrossEntropyLoss()

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        **kwargs,
    ):
        loss = 0

        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )

        query_document_embedding = outputs.pooler_output

        if labels is not None:
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, query_document_embedding

    def get_mlm_loss(
        self,
        sequence_output: FloatTensor,
        labels: LongTensor,
    ):
        token_scores = self.mlm_head(sequence_output)

        return self.mlm_loss(
            token_scores.view(-1, self.config.vocab_size),
            labels.view(-1),
        )


class CrossEncoder(BertModel):
    """
    BERT cross-encoder for Baidu-ULTR.

    The model uses a shared BERT encoder with task-specific heads for
    click, skip, and dwell-time prediction.

    Position bias can be:
        - disabled,
        - shared between click and skip, or
        - modeled separately for click, skip, and dwell time.

    Args:
        config:
            Hugging Face BERT configuration.

        multi_task:
            If True, use click, skip, and dwell-time tasks.
            If False, use only the click task.

        debiasing:
            Whether position-bias correction is applied during training.

        separate_bias_per_task:
            If False, the bias tower outputs one position-bias logit that
            is shared by click and skip. Dwell time receives no additional
            position correction.

            If True, the bias tower outputs three logits corresponding to
            click, skip, and dwell time.

        dwell_loss_weight:
            Weight applied to the dwell-time loss.

        position_vocab_size:
            Number of supported ranking positions.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        multi_task: bool = True,
        debiasing: bool = True,
        separate_bias_per_task: bool = False,
        dwell_loss_weight: float = 0.1,
        position_vocab_size: int = 30,
    ):
        super(CrossEncoder, self).__init__(config)

        # Task-specific relevance heads.
        self.click_head = nn.Linear(config.hidden_size, 1)
        self.skip_head = nn.Linear(config.hidden_size, 1)
        self.dwell_time_head = nn.Linear(config.hidden_size, 1)

        # Task losses.
        self.click_loss = nn.BCEWithLogitsLoss()
        self.skip_loss = nn.BCEWithLogitsLoss()
        self.dwell_time_loss = nn.MSELoss(reduction="none")

        # Position-bias tower.
        d_model = 1
        num_logits = 3 if separate_bias_per_task else 1

        self.bias_tower = nn.Sequential(
            nn.Embedding(position_vocab_size, d_model),
            nn.Linear(d_model, num_logits),
        )

        # Model configuration.
        self.multi_task = multi_task
        self.debiasing = debiasing
        self.separate_bias_per_task = separate_bias_per_task
        self.dwell_loss_weight = dwell_loss_weight
        self.position_vocab_size = position_vocab_size


    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        clicks: Optional[FloatTensor] = None,
        skips: Optional[FloatTensor] = None,
        dwell_time: Optional[FloatTensor] = None,
        positions: Optional[LongTensor] = None,
        **kwargs,
    ):
        loss_per_task = {}

        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )

        query_document_embedding = outputs.pooler_output

        # Position bias

        bias_scores = None

        if positions is not None:
            bias_scores = self.bias_tower(positions)

            # Preserve the original architecture while effectively
            # disabling the bias contribution.
            if not self.debiasing:
                bias_scores = torch.zeros_like(bias_scores)

        # Click task

        if clicks is not None:
            if bias_scores is None:
                raise ValueError(
                    "positions must be provided when computing click loss."
                )

            click_scores = self.click_head(query_document_embedding)

            if bias_scores.size(1) == 1:
                click_scores = click_scores + bias_scores
            else:
                click_scores = (
                    click_scores
                    + bias_scores[:, 0].unsqueeze(-1)
                )

            click_loss = self.click_loss(
                click_scores.squeeze(-1),
                clicks,
            )

            loss_per_task["click"] = click_loss

        # Multi-task objectives

        if self.multi_task:

            # Skip task.
            if skips is not None:
                if bias_scores is None:
                    raise ValueError(
                        "positions must be provided when computing skip loss."
                    )

                skip_scores = self.skip_head(
                    query_document_embedding
                )

                if bias_scores.size(1) == 1:
                    skip_scores = skip_scores + bias_scores
                else:
                    skip_scores = (
                        skip_scores
                        + bias_scores[:, 1].unsqueeze(-1)
                    )

                skip_loss = self.skip_loss(
                    skip_scores.squeeze(-1),
                    skips,
                )

                loss_per_task["skip"] = skip_loss

            # Dwell-time task.
            if dwell_time is not None and clicks is not None:
                dwell_time_scores = self.dwell_time_head(
                    query_document_embedding
                )

                # For per-task debiasing, dwell time receives its own
                # position-bias logit.
                # For shared debiasing, position bias is applied only
                # to click and skip.
                if (
                    self.debiasing
                    and self.separate_bias_per_task
                    and bias_scores is not None
                ):
                    dwell_time_scores = (
                        dwell_time_scores
                        + bias_scores[:, 2].unsqueeze(-1)
                    )

                dwell_time_scores = dwell_time_scores.squeeze(-1)

                per_example_loss = self.dwell_time_loss(
                    dwell_time_scores,
                    dwell_time,
                )

                # Dwell time is observed only for clicked documents.
                dwell_time_loss = (
                    (clicks * per_example_loss).sum()
                    / clicks.sum().clamp(min=1)
                )

                loss_per_task["dwell_time"] = (
                    self.dwell_loss_weight * dwell_time_loss
                )

        # Masked-language modeling

        if labels is not None:
            mlm_loss = self.get_mlm_loss(
                outputs[0],
                labels,
            )

            loss_per_task["mlm"] = mlm_loss

        return loss_per_task, query_document_embedding

    @torch.no_grad()
    def predict(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        **kwargs,
    ):
        """
        Produce relevance-head predictions without position-bias logits.

        This is the inference setup used on the expert-annotated
        Baidu-ULTR evaluation data.
        """

        self.eval()

        _, query_document_embedding = self.forward(
            tokens=tokens,
            attention_mask=attention_mask,
            token_types=token_types,
        )

        click_scores = (
            self.click_head(query_document_embedding)
            .squeeze(-1)
            .cpu()
            .tolist()
        )

        if not self.multi_task:
            return click_scores

        skip_scores = (
            self.skip_head(query_document_embedding)
            .squeeze(-1)
            .cpu()
            .tolist()
        )

        dwell_time_scores = (
            self.dwell_time_head(query_document_embedding)
            .squeeze(-1)
            .cpu()
            .tolist()
        )

        return (
            click_scores,
            skip_scores,
            dwell_time_scores,
        )
